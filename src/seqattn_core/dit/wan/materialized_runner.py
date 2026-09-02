from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import ProjectedAttentionRunner, ProjectedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    MaterializedAttentionExecutor,
    TiledHostStageRunner,
    validate_hidden_host,
)
from .stats import WanDiTStats
from .types import (
    WanBlockOps,
    WanMaterializedProjections,
    WanSequenceMeta,
    validate_wan_runner_contract,
)


class WanMaterializedRunner:
    """Wan block order: self-attention, text cross-attention, then FFN."""

    def __init__(
        self,
        self_attention: ProjectedAttentionRunner,
        cross_attention: ProjectedCrossAttentionRunner,
        *,
        hidden_features: int,
        ffn_tile_tokens: int,
        num_output_buffers: int = 2,
    ) -> None:
        init_single_flight(self)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.hidden_features = hidden_features
        validate_wan_runner_contract(
            self_attention,
            cross_attention,
            hidden_features=hidden_features,
        )
        plan = self_attention.plan
        output_chunk_tokens = max(plan.q_chunk_tokens, cross_attention.plan.q_chunk_tokens)
        self.output_workspace = AttentionOutputWorkspace(
            hidden_features=hidden_features,
            output_chunk_tokens=output_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
            num_output_buffers=num_output_buffers,
        )
        self.consumer = AttentionOutputConsumer(self.output_workspace)
        self.attention_executor = MaterializedAttentionExecutor(self.consumer)
        self.ffn = TiledHostStageRunner(
            hidden_features=hidden_features,
            chunk_tokens=ffn_tile_tokens,
            dtype=plan.dtype,
            device=plan.device,
            require_pinned_hidden=self_attention.pipeline_config.require_pinned_hidden,
        )

    def _validate_inputs(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
    ) -> None:
        validate_hidden_host(
            hidden_host,
            plan=self.self_attention.plan,
            hidden_features=self.hidden_features,
            require_pinned=self.self_attention.pipeline_config.require_pinned_hidden,
        )
        self.cross_attention.validate_inputs(hidden_host, text_hidden_host)
        sequence_meta.validate(hidden_host.shape[0], text_hidden_host.shape[0])

    @single_flight
    @torch.inference_mode()
    def run_block_(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        projections: WanMaterializedProjections,
        ops: WanBlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_host, text_hidden_host, sequence_meta)
        stats = WanDiTStats() if stats is None else stats
        stats.backend = self.self_attention.attention.backend
        stats.qkv_storage_policy = "materialized"
        started = time.perf_counter()

        self.attention_executor.run_self(
            self.self_attention,
            hidden_host,
            sequence_meta.hidden_cu_seqlens,
            projections.self_attention,
            epilogue=ops.self_attention_epilogue,
            consumer_lease=ops.self_attention_lease,
            softmax_scale=self_softmax_scale,
            causal=self_causal,
            stats=stats.self_attention,
        )
        self.attention_executor.run_cross(
            self.cross_attention,
            hidden_host,
            text_hidden_host,
            sequence_meta.hidden_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.text_cross_attention,
            epilogue=ops.cross_attention_epilogue,
            consumer_lease=ops.cross_attention_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.cross_attention,
        )
        arenas = {
            id(runner.arena): runner.arena for runner in (self.self_attention, self.cross_attention)
        }
        stats.qkv_host_bytes_peak = max(
            stats.qkv_host_bytes_peak,
            sum(arena.allocated_bytes for arena in arenas.values()),
        )

        with ops.ffn_context():
            self.ffn.run(hidden_host, hidden_host, ops.ffn, stats=stats.ffn)
        stats.hidden_host_bytes_peak = max(
            stats.hidden_host_bytes_peak,
            (hidden_host.numel() + text_hidden_host.numel()) * hidden_host.element_size(),
        )
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        blocks: Iterable[tuple[WanMaterializedProjections, WanBlockOps]],
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        stats = WanDiTStats() if stats is None else stats
        for projections, ops in blocks:
            self.run_block_(
                hidden_host,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                self_softmax_scale=self_softmax_scale,
                cross_softmax_scale=cross_softmax_scale,
                self_causal=self_causal,
                stats=stats,
            )
        return hidden_host


__all__ = ["WanMaterializedRunner"]
