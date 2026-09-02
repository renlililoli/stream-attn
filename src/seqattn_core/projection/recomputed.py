from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext, suppress

import torch

from .._single_flight import init_single_flight, single_flight
from ..plan import AttentionPlan
from ..stats import RecomputedAttentionStats, RecomputedCrossAttentionStats
from ..streaming import StreamingAttentionRunner
from ..streaming.protocols import DeviceOutputConsumer
from ..streaming.tile_source import (
    QKVTileSource,
    RecomputedCrossQKVTileSource,
    RecomputedQKVTileSource,
)
from .contracts import KVTileProjector, QTileProjector
from .memory import CrossRecomputeWorkspace, RecomputeWorkspace
from .validation import validate_projection_hidden

RecomputedStats = RecomputedAttentionStats | RecomputedCrossAttentionStats


class _RecomputedAttentionBase:
    def __init__(self, plan: AttentionPlan, *, require_pinned_hidden: bool) -> None:
        init_single_flight(self)
        if plan.output_mode != "device_consumer":
            raise ValueError("recomputed attention requires device_consumer output mode")
        if plan.device.type != "cuda":
            raise ValueError("recomputed attention requires a CUDA device")
        self.plan = plan
        self.attention = StreamingAttentionRunner(plan)
        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for recomputed attention")
        self.require_pinned_hidden = require_pinned_hidden

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.plan.enable_nvtx else nullcontext()

    def _run_source(
        self,
        source: QKVTileSource,
        *,
        q_tokens: int,
        kv_tokens: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        output_consumer: DeviceOutputConsumer,
        softmax_scale: float | None,
        causal: bool,
        stats: RecomputedStats,
        range_name: str,
    ) -> None:
        def execute(source: QKVTileSource) -> None:
            self.attention.run_with_qkv_source(
                source,
                q_tokens,
                kv_tokens,
                cu_seqlens_q,
                cu_seqlens_kv,
                output_consumer=output_consumer,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats.attention,
            )

        self._run_source_executor(
            source,
            execute=execute,
            stats=stats,
            range_name=range_name,
        )

    def _run_source_executor(
        self,
        source: QKVTileSource,
        *,
        execute: Callable[[QKVTileSource], None],
        stats: RecomputedStats,
        range_name: str,
    ) -> None:
        started = time.perf_counter()
        try:
            with self._range(range_name):
                execute(source)
        except Exception:
            with suppress(Exception):
                torch.cuda.synchronize(self.plan.device)
            source.recover()
            raise
        stats.wall_seconds += time.perf_counter() - started

    def _record_common_stats(
        self,
        stats: RecomputedStats,
        q_tokens: int,
        element_size: int,
    ) -> None:
        stats.backend = self.attention.backend
        stats.qkv_host_bytes = 0
        stats.raw_attention_roundtrip_bytes_avoided += (
            2 * q_tokens * self.plan.q_heads * self.plan.head_dim * element_size
        )


class RecomputedAttentionRunner(_RecomputedAttentionBase):
    """Exact self-attention with Q and K/V projected directly into device tiles."""

    def __init__(
        self,
        plan: AttentionPlan,
        *,
        hidden_features: int,
        require_pinned_hidden: bool = True,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        super().__init__(plan, require_pinned_hidden=require_pinned_hidden)
        self.hidden_features = hidden_features
        self.workspace = RecomputeWorkspace(
            hidden_features=hidden_features,
            staging_tokens=max(plan.q_chunk_tokens, plan.kv_chunk_tokens),
            dtype=plan.dtype,
            device=plan.device,
        )

    @property
    def hidden_staging_bytes(self) -> int:
        return self.workspace.hidden.numel() * self.workspace.hidden.element_size()

    def validate_hidden(self, hidden_cpu: torch.Tensor) -> None:
        validate_projection_hidden(
            hidden_cpu,
            plan=self.plan,
            require_pinned=self.require_pinned_hidden,
            hidden_features=self.hidden_features,
        )

    @single_flight
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
        def execute(source: QKVTileSource) -> None:
            self.attention.run_with_qkv_source(
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

        stats = RecomputedAttentionStats() if stats is None else stats
        self._run_with_source_executor(
            hidden_cpu,
            project_q=project_q,
            project_kv=project_kv,
            execute=execute,
            stats=stats,
            range_name="seqattn:recomputed_attention",
        )

    @single_flight
    @torch.inference_mode()
    def run_with_source_executor(
        self,
        hidden_cpu: torch.Tensor,
        *,
        project_q: QTileProjector,
        project_kv: KVTileProjector,
        execute: Callable[[QKVTileSource], None],
        stats: RecomputedAttentionStats | None = None,
        range_name: str = "seqattn:recomputed_attention",
    ) -> None:
        stats = RecomputedAttentionStats() if stats is None else stats
        self._run_with_source_executor(
            hidden_cpu,
            project_q=project_q,
            project_kv=project_kv,
            execute=execute,
            stats=stats,
            range_name=range_name,
        )

    def _run_with_source_executor(
        self,
        hidden_cpu: torch.Tensor,
        *,
        project_q: QTileProjector,
        project_kv: KVTileProjector,
        execute: Callable[[QKVTileSource], None],
        stats: RecomputedAttentionStats,
        range_name: str,
    ) -> None:
        self.validate_hidden(hidden_cpu)
        self._record_common_stats(stats, hidden_cpu.shape[0], hidden_cpu.element_size())
        source = RecomputedQKVTileSource(
            hidden_cpu,
            self.workspace,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
            enable_nvtx=self.plan.enable_nvtx,
        )
        self._run_source_executor(
            source,
            execute=execute,
            stats=stats,
            range_name=range_name,
        )


class RecomputedCrossAttentionRunner(_RecomputedAttentionBase):
    """Exact cross-attention with Q and K/V projected directly into device tiles."""

    def __init__(
        self,
        plan: AttentionPlan,
        *,
        query_hidden_features: int,
        context_hidden_features: int,
        require_pinned_hidden: bool = True,
    ) -> None:
        if query_hidden_features <= 0 or context_hidden_features <= 0:
            raise ValueError("query and context hidden feature sizes must be positive")
        super().__init__(plan, require_pinned_hidden=require_pinned_hidden)
        self.query_hidden_features = query_hidden_features
        self.context_hidden_features = context_hidden_features
        self.workspace = CrossRecomputeWorkspace(
            query_hidden_features=query_hidden_features,
            context_hidden_features=context_hidden_features,
            q_staging_tokens=plan.q_chunk_tokens,
            kv_staging_tokens=plan.kv_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
        )

    @property
    def hidden_staging_bytes(self) -> int:
        tensors = (self.workspace.query.hidden, self.workspace.context.hidden)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def validate_hidden(
        self,
        hidden_host: torch.Tensor,
        *,
        hidden_features: int,
        max_tokens: int,
        name: str,
    ) -> None:
        validate_projection_hidden(
            hidden_host,
            plan=self.plan,
            require_pinned=self.require_pinned_hidden,
            hidden_features=hidden_features,
            max_tokens=max_tokens,
            name=name,
        )

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
        output_consumer: DeviceOutputConsumer,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: RecomputedCrossAttentionStats | None = None,
    ) -> None:
        self.validate_inputs(query_hidden_host, context_hidden_host)
        stats = RecomputedCrossAttentionStats() if stats is None else stats
        self._record_common_stats(
            stats,
            query_hidden_host.shape[0],
            query_hidden_host.element_size(),
        )
        source = RecomputedCrossQKVTileSource(
            query_hidden_host,
            context_hidden_host,
            self.workspace,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
            enable_nvtx=self.plan.enable_nvtx,
        )
        self._run_source(
            source,
            q_tokens=query_hidden_host.shape[0],
            kv_tokens=context_hidden_host.shape[0],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            output_consumer=output_consumer,
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats,
            range_name="seqattn:recomputed_cross_attention",
        )


__all__ = ["RecomputedAttentionRunner", "RecomputedCrossAttentionRunner"]
