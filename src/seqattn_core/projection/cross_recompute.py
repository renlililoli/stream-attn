from __future__ import annotations

import time
from contextlib import nullcontext, suppress

import torch

from .._single_flight import init_single_flight, single_flight
from ..config import StreamingAttentionConfig
from ..planner import AttentionPlan
from ..stats import RecomputedCrossAttentionStats
from ..streaming import StreamingAttentionRunner
from ..streaming.tile_source import RecomputedCrossQKVTileSource
from .recompute_workspace import CrossRecomputeWorkspace
from .types import KVTileProjector, QTileProjector


class RecomputedCrossAttentionRunner:
    """Exact cross-attention with Q and K/V projected directly into device tiles."""

    def __init__(
        self,
        plan: AttentionPlan,
        *,
        query_hidden_features: int,
        context_hidden_features: int,
        attention_config: StreamingAttentionConfig | None = None,
        require_pinned_hidden: bool = True,
        enable_nvtx: bool = False,
    ) -> None:
        init_single_flight(self)
        if query_hidden_features <= 0 or context_hidden_features <= 0:
            raise ValueError("query and context hidden feature sizes must be positive")
        if plan.output_mode != "device_consumer":
            raise ValueError("recomputed cross-attention requires device_consumer output mode")
        self.plan = plan
        if self.plan.device.type != "cuda":
            raise ValueError("recomputed cross-attention requires a CUDA device")
        self.attention = StreamingAttentionRunner(self.plan, attention_config)
        self.attention_config = self.attention.config
        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for recomputed cross-attention")
        self.query_hidden_features = query_hidden_features
        self.context_hidden_features = context_hidden_features
        self.require_pinned_hidden = require_pinned_hidden
        self.enable_nvtx = enable_nvtx
        self.workspace = CrossRecomputeWorkspace(
            query_hidden_features=query_hidden_features,
            context_hidden_features=context_hidden_features,
            q_staging_tokens=self.plan.q_chunk_tokens,
            kv_staging_tokens=self.plan.kv_chunk_tokens,
            dtype=self.plan.dtype,
            device=self.plan.device,
        )

    @property
    def hidden_staging_bytes(self) -> int:
        tensors = (self.workspace.query.hidden, self.workspace.context.hidden)
        return sum(t.numel() * t.element_size() for t in tensors)

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.enable_nvtx else nullcontext()

    def _recover_after_failure(self) -> None:
        with suppress(Exception):
            torch.cuda.synchronize(self.plan.device)
        self.workspace.query.hidden_has_pending_compute = False
        self.workspace.context.hidden_has_pending_compute = False

    def validate_hidden(
        self,
        hidden_host: torch.Tensor,
        *,
        hidden_features: int,
        max_tokens: int,
        name: str,
    ) -> None:
        if hidden_host.device.type != "cpu" or hidden_host.ndim != 2:
            raise ValueError(f"{name} must use CPU [tokens, hidden_features] layout")
        if hidden_host.shape[1] != hidden_features:
            raise ValueError(f"{name} feature size does not match the recompute runner")
        if hidden_host.shape[0] > max_tokens:
            raise ValueError(f"{name} token count exceeds the runner plan")
        if hidden_host.dtype != self.plan.dtype:
            raise ValueError(f"{name} dtype must match the attention plan")
        if not hidden_host.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if self.require_pinned_hidden and not hidden_host.is_pinned():
            raise ValueError(f"asynchronous recompute requires pinned {name}")

    def validate_inputs(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
    ) -> None:
        self.validate_hidden(
            query_hidden_host,
            hidden_features=self.query_hidden_features,
            max_tokens=self.plan.max_q_tokens,
            name="query_hidden_host",
        )
        self.validate_hidden(
            context_hidden_host,
            hidden_features=self.context_hidden_features,
            max_tokens=self.plan.max_kv_tokens,
            name="context_hidden_host",
        )

    @single_flight
    @torch.inference_mode()
    def run_with_device_consumer(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        *,
        project_q: QTileProjector,
        project_kv: KVTileProjector,
        output_consumer,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: RecomputedCrossAttentionStats | None = None,
    ) -> None:
        self.validate_inputs(query_hidden_host, context_hidden_host)
        stats = RecomputedCrossAttentionStats() if stats is None else stats
        stats.backend = self.attention.backend
        stats.qkv_host_bytes = 0
        stats.raw_attention_roundtrip_bytes_avoided += (
            2
            * query_hidden_host.shape[0]
            * self.plan.q_heads
            * self.plan.head_dim
            * query_hidden_host.element_size()
        )
        source = RecomputedCrossQKVTileSource(
            query_hidden_host,
            context_hidden_host,
            self.workspace,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
            enable_nvtx=self.enable_nvtx,
        )
        started = time.perf_counter()
        try:
            with self._range("seqattn:recomputed_cross_attention"):
                self.attention.run_with_qkv_source(
                    source,
                    query_hidden_host.shape[0],
                    context_hidden_host.shape[0],
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    output_consumer=output_consumer,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    stats=stats.attention,
                )
        except Exception:
            self._recover_after_failure()
            raise
        stats.wall_seconds += time.perf_counter() - started


__all__ = ["RecomputedCrossAttentionRunner"]
