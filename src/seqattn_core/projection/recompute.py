from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from ..config import StreamingAttentionConfig
from ..planner import AttentionPlan, build_plan
from ..stats import RecomputedAttentionStats
from ..streaming import StreamingAttentionRunner
from ..streaming.protocols import DeviceOutputConsumer
from ..streaming.tile_source import RecomputedQKVTileSource
from .recompute_workspace import RecomputeWorkspace
from .types import KVTileProjector, QTileProjector


class RecomputedAttentionRunner:
    """Exact attention with Q and K/V projected directly into device tiles."""

    def __init__(
        self,
        plan: AttentionPlan,
        *,
        hidden_features: int,
        attention_config: StreamingAttentionConfig | None = None,
        require_pinned_hidden: bool = True,
        enable_nvtx: bool = False,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        self.attention_config = (
            StreamingAttentionConfig(output_mode="device_consumer")
            if attention_config is None
            else attention_config
        )
        self.attention_config.validate()
        if self.attention_config.output_mode != "device_consumer":
            raise ValueError("recomputed attention requires device_consumer output mode")
        self.plan = build_plan(
            q_heads=plan.q_heads,
            kv_heads=plan.kv_heads,
            head_dim=plan.head_dim,
            dtype=plan.dtype,
            device=plan.device,
            max_q_tokens=plan.max_q_tokens,
            max_kv_tokens=plan.max_kv_tokens,
            config=self.attention_config,
        )
        if self.plan.device.type != "cuda":
            raise ValueError("recomputed attention requires a CUDA device")

        self.attention = StreamingAttentionRunner(self.plan, self.attention_config)
        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for recomputed attention")
        self.hidden_features = hidden_features
        self.require_pinned_hidden = require_pinned_hidden
        self.enable_nvtx = enable_nvtx
        self.workspace = RecomputeWorkspace(
            hidden_features=hidden_features,
            staging_tokens=max(self.plan.q_chunk_tokens, self.plan.kv_chunk_tokens),
            dtype=self.plan.dtype,
            device=self.plan.device,
        )

    @property
    def hidden_staging_bytes(self) -> int:
        return self.workspace.hidden.numel() * self.workspace.hidden.element_size()

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.enable_nvtx else nullcontext()

    def _validate_hidden(self, hidden_cpu: torch.Tensor) -> None:
        if hidden_cpu.device.type != "cpu" or hidden_cpu.ndim != 2:
            raise ValueError("hidden_cpu must use CPU [tokens, hidden_features] layout")
        if hidden_cpu.shape[1] != self.hidden_features:
            raise ValueError("hidden_cpu feature size does not match the recompute runner")
        if hidden_cpu.dtype != self.plan.dtype:
            raise ValueError("hidden_cpu dtype must match the attention plan")
        if not hidden_cpu.is_contiguous():
            raise ValueError("hidden_cpu must be contiguous")
        if self.require_pinned_hidden and not hidden_cpu.is_pinned():
            raise ValueError("asynchronous recompute requires pinned hidden_cpu")
        tokens = hidden_cpu.shape[0]
        if tokens > self.plan.max_q_tokens or tokens > self.plan.max_kv_tokens:
            raise ValueError("hidden token count exceeds the runner plan")

    @torch.inference_mode()
    def run_with_device_consumer(
        self,
        hidden_cpu: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        project_q: QTileProjector,
        project_kv: KVTileProjector,
        output_consumer: DeviceOutputConsumer,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: RecomputedAttentionStats | None = None,
    ) -> None:
        self._validate_hidden(hidden_cpu)
        stats = RecomputedAttentionStats() if stats is None else stats
        stats.backend = self.attention.backend
        stats.qkv_host_bytes = 0
        source = RecomputedQKVTileSource(
            hidden_cpu,
            self.workspace,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
            enable_nvtx=self.enable_nvtx,
        )
        started = time.perf_counter()
        with self._range("seqattn:recomputed_attention"):
            self.attention._run_with_qkv_source(
                source,
                hidden_cpu.shape[0],
                hidden_cpu.shape[0],
                cu_seqlens,
                cu_seqlens,
                output_consumer=output_consumer,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats.attention,
            )
        elapsed = time.perf_counter() - started
        stats.wall_seconds += elapsed


__all__ = ["RecomputedAttentionRunner"]
