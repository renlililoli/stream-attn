from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import RecomputedAttentionRunner, RecomputedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    RecomputedAttentionExecutor,
    TiledHostStageRunner,
    require_distinct_storage,
    validate_hidden_host,
)
from .stats import WanDiTStats
from .types import (
    WanBlockOps,
    WanRecomputeProjections,
    WanSequenceMeta,
    validate_wan_runner_contract,
)


class WanRecomputeRunner:
    """Wan block runner without sequence-sized host Q/K/V tensors."""

    def __init__(
        self,
        self_attention: RecomputedAttentionRunner,
        cross_attention: RecomputedCrossAttentionRunner,
        *,
        ffn_tile_tokens: int,
        num_output_buffers: int = 2,
    ) -> None:
        init_single_flight(self)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.hidden_features = self_attention.hidden_features
        validate_wan_runner_contract(
            self_attention,
            cross_attention,
            hidden_features=self.hidden_features,
        )
        plan = self_attention.plan
        output_chunk_tokens = max(plan.q_chunk_tokens, cross_attention.plan.q_chunk_tokens)
        self.output_workspace = AttentionOutputWorkspace(
            hidden_features=self.hidden_features,
            output_chunk_tokens=output_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
            num_output_buffers=num_output_buffers,
        )
        self.consumer = AttentionOutputConsumer(self.output_workspace)
        self.attention_executor = RecomputedAttentionExecutor(self.consumer)
        self.ffn = TiledHostStageRunner(
            hidden_features=self.hidden_features,
            chunk_tokens=ffn_tile_tokens,
            dtype=plan.dtype,
            device=plan.device,
            require_pinned_hidden=self_attention.require_pinned_hidden,
        )

    def _validate_inputs(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
    ) -> None:
        for name, tensor in (
            ("source_hidden_host", source_hidden_host),
            ("destination_hidden_host", destination_hidden_host),
        ):
            validate_hidden_host(
                tensor,
                plan=self.self_attention.plan,
                hidden_features=self.hidden_features,
                require_pinned=self.self_attention.require_pinned_hidden,
                name=name,
            )
        require_distinct_storage(source_hidden_host, destination_hidden_host)
        self.cross_attention.validate_hidden(
            text_hidden_host,
            hidden_features=self.cross_attention.context_hidden_features,
            max_tokens=self.cross_attention.plan.max_kv_tokens,
            name="text_hidden_host",
        )
        sequence_meta.validate(source_hidden_host.shape[0], text_hidden_host.shape[0])

    @single_flight
    @torch.inference_mode()
    def run_block(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        projections: WanRecomputeProjections,
        ops: WanBlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            source_hidden_host,
            destination_hidden_host,
            text_hidden_host,
            sequence_meta,
        )
        stats = WanDiTStats() if stats is None else stats
        stats.backend = self.self_attention.attention.backend
        stats.qkv_storage_policy = "recompute"
        started = time.perf_counter()

        self.attention_executor.run_self(
            self.self_attention,
            source_hidden_host,
            destination_hidden_host,
            sequence_meta.hidden_cu_seqlens,
            projections.self_attention,
            epilogue=ops.self_attention_epilogue,
            consumer_lease=ops.self_attention_lease,
            softmax_scale=self_softmax_scale,
            causal=self_causal,
            stats=stats.self_recompute,
        )
        self.attention_executor.run_cross(
            self.cross_attention,
            destination_hidden_host,
            text_hidden_host,
            destination_hidden_host,
            sequence_meta.hidden_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.text_cross_attention,
            epilogue=ops.cross_attention_epilogue,
            consumer_lease=ops.cross_attention_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.cross_recompute,
        )

        with ops.ffn_context():
            self.ffn.run(
                destination_hidden_host,
                destination_hidden_host,
                ops.ffn,
                stats=stats.ffn,
            )
        hidden_bytes = (
            source_hidden_host.numel() + destination_hidden_host.numel() + text_hidden_host.numel()
        ) * source_hidden_host.element_size()
        stats.hidden_host_bytes_peak = max(stats.hidden_host_bytes_peak, hidden_bytes)
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return destination_hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        scratch_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        blocks: Iterable[tuple[WanRecomputeProjections, WanBlockOps]],
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        stats = WanDiTStats() if stats is None else stats
        source = hidden_host
        destination = scratch_hidden_host
        for projections, ops in blocks:
            self.run_block(
                source,
                destination,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                self_softmax_scale=self_softmax_scale,
                cross_softmax_scale=cross_softmax_scale,
                self_causal=self_causal,
                stats=stats,
            )
            source, destination = destination, source
        return source


__all__ = ["WanRecomputeRunner"]
