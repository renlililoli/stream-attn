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
from .stats import LTX2DiTStats
from .types import (
    LTX2BlockOps,
    LTX2RecomputeProjections,
    LTX2SequenceMeta,
    validate_ltx2_runner_contract,
)


class LTX2RecomputeRunner:
    """LTX2 block runner without sequence-sized host Q/K/V storage."""

    def __init__(
        self,
        *,
        video_self_attention: RecomputedAttentionRunner,
        audio_self_attention: RecomputedAttentionRunner,
        video_text_attention: RecomputedCrossAttentionRunner,
        audio_text_attention: RecomputedCrossAttentionRunner,
        video_from_audio_attention: RecomputedCrossAttentionRunner,
        audio_from_video_attention: RecomputedCrossAttentionRunner,
        video_ffn_tile_tokens: int,
        audio_ffn_tile_tokens: int,
        num_output_buffers: int = 2,
    ) -> None:
        init_single_flight(self)
        self.video_self_attention = video_self_attention
        self.audio_self_attention = audio_self_attention
        self.video_text_attention = video_text_attention
        self.audio_text_attention = audio_text_attention
        self.video_from_audio_attention = video_from_audio_attention
        self.audio_from_video_attention = audio_from_video_attention
        self.video_hidden_features = video_self_attention.hidden_features
        self.audio_hidden_features = audio_self_attention.hidden_features
        validate_ltx2_runner_contract(
            video_self_attention=video_self_attention,
            audio_self_attention=audio_self_attention,
            video_text_attention=video_text_attention,
            audio_text_attention=audio_text_attention,
            video_from_audio_attention=video_from_audio_attention,
            audio_from_video_attention=audio_from_video_attention,
            video_hidden_features=self.video_hidden_features,
            audio_hidden_features=self.audio_hidden_features,
        )
        video_plan = video_self_attention.plan
        audio_plan = audio_self_attention.plan
        self.video_consumer = AttentionOutputConsumer(
            AttentionOutputWorkspace(
                hidden_features=self.video_hidden_features,
                output_chunk_tokens=max(
                    video_plan.q_chunk_tokens,
                    video_text_attention.plan.q_chunk_tokens,
                    video_from_audio_attention.plan.q_chunk_tokens,
                ),
                dtype=video_plan.dtype,
                device=video_plan.device,
                num_output_buffers=num_output_buffers,
            )
        )
        self.audio_consumer = AttentionOutputConsumer(
            AttentionOutputWorkspace(
                hidden_features=self.audio_hidden_features,
                output_chunk_tokens=max(
                    audio_plan.q_chunk_tokens,
                    audio_text_attention.plan.q_chunk_tokens,
                    audio_from_video_attention.plan.q_chunk_tokens,
                ),
                dtype=audio_plan.dtype,
                device=audio_plan.device,
                num_output_buffers=num_output_buffers,
            )
        )
        self.video_attention_executor = RecomputedAttentionExecutor(self.video_consumer)
        self.audio_attention_executor = RecomputedAttentionExecutor(self.audio_consumer)
        self.video_ffn = TiledHostStageRunner(
            hidden_features=self.video_hidden_features,
            chunk_tokens=video_ffn_tile_tokens,
            dtype=video_plan.dtype,
            device=video_plan.device,
            require_pinned_hidden=video_self_attention.require_pinned_hidden,
        )
        self.audio_ffn = TiledHostStageRunner(
            hidden_features=self.audio_hidden_features,
            chunk_tokens=audio_ffn_tile_tokens,
            dtype=audio_plan.dtype,
            device=audio_plan.device,
            require_pinned_hidden=audio_self_attention.require_pinned_hidden,
        )

    def _validate_inputs(
        self,
        video_source: torch.Tensor,
        video_destination: torch.Tensor,
        audio_source: torch.Tensor,
        audio_destination: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
    ) -> None:
        for tensor, name, runner, features in (
            (video_source, "video_source", self.video_self_attention, self.video_hidden_features),
            (
                video_destination,
                "video_destination",
                self.video_self_attention,
                self.video_hidden_features,
            ),
            (audio_source, "audio_source", self.audio_self_attention, self.audio_hidden_features),
            (
                audio_destination,
                "audio_destination",
                self.audio_self_attention,
                self.audio_hidden_features,
            ),
        ):
            validate_hidden_host(
                tensor,
                plan=runner.plan,
                hidden_features=features,
                require_pinned=runner.require_pinned_hidden,
                name=name,
            )
        require_distinct_storage(video_source, video_destination)
        require_distinct_storage(audio_source, audio_destination)
        self.video_text_attention.validate_inputs(video_source, text_hidden_host)
        self.audio_text_attention.validate_inputs(audio_source, text_hidden_host)
        sequence_meta.validate(
            video_source.shape[0],
            audio_source.shape[0],
            text_hidden_host.shape[0],
        )

    @single_flight
    @torch.inference_mode()
    def run_block(
        self,
        video_source: torch.Tensor,
        video_destination: torch.Tensor,
        audio_source: torch.Tensor,
        audio_destination: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
        projections: LTX2RecomputeProjections,
        ops: LTX2BlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        stats: LTX2DiTStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            video_source,
            video_destination,
            audio_source,
            audio_destination,
            text_hidden_host,
            sequence_meta,
        )
        stats = LTX2DiTStats() if stats is None else stats
        stats.backend = self.video_self_attention.attention.backend
        stats.qkv_storage_policy = "recompute"
        started = time.perf_counter()

        self.video_attention_executor.run_self(
            self.video_self_attention,
            video_source,
            video_destination,
            sequence_meta.video_cu_seqlens,
            projections.video_self_attention,
            epilogue=ops.video_self_attention.epilogue,
            consumer_lease=ops.video_self_attention.weight_lease,
            softmax_scale=self_softmax_scale,
            causal=False,
            stats=stats.video_self_recompute,
        )
        self.audio_attention_executor.run_self(
            self.audio_self_attention,
            audio_source,
            audio_destination,
            sequence_meta.audio_cu_seqlens,
            projections.audio_self_attention,
            epilogue=ops.audio_self_attention.epilogue,
            consumer_lease=ops.audio_self_attention.weight_lease,
            softmax_scale=self_softmax_scale,
            causal=False,
            stats=stats.audio_self_recompute,
        )
        self.video_attention_executor.run_cross(
            self.video_text_attention,
            video_destination,
            text_hidden_host,
            video_source,
            sequence_meta.video_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.video_text_attention,
            epilogue=ops.video_text_attention.epilogue,
            consumer_lease=ops.video_text_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.video_text_recompute,
        )
        self.audio_attention_executor.run_cross(
            self.audio_text_attention,
            audio_destination,
            text_hidden_host,
            audio_source,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.audio_text_attention,
            epilogue=ops.audio_text_attention.epilogue,
            consumer_lease=ops.audio_text_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.audio_text_recompute,
        )

        # Both directions read the same post-text snapshots and write alternate buffers.
        self.video_attention_executor.run_cross(
            self.video_from_audio_attention,
            video_source,
            audio_source,
            video_destination,
            sequence_meta.video_cu_seqlens,
            sequence_meta.audio_cu_seqlens,
            projections.video_from_audio_attention,
            epilogue=ops.video_from_audio_attention.epilogue,
            consumer_lease=ops.video_from_audio_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.video_from_audio_recompute,
        )
        self.audio_attention_executor.run_cross(
            self.audio_from_video_attention,
            audio_source,
            video_source,
            audio_destination,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.video_cu_seqlens,
            projections.audio_from_video_attention,
            epilogue=ops.audio_from_video_attention.epilogue,
            consumer_lease=ops.audio_from_video_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.audio_from_video_recompute,
        )
        with ops.video_ffn.context():
            self.video_ffn.run(
                video_destination,
                video_destination,
                ops.video_ffn.operation,
                stats=stats.video_ffn,
            )
        with ops.audio_ffn.context():
            self.audio_ffn.run(
                audio_destination,
                audio_destination,
                ops.audio_ffn.operation,
                stats=stats.audio_ffn,
            )
        hidden_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                video_source,
                video_destination,
                audio_source,
                audio_destination,
                text_hidden_host,
            )
        )
        stats.hidden_host_bytes_peak = max(stats.hidden_host_bytes_peak, hidden_bytes)
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return video_destination, audio_destination

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        video_hidden_host: torch.Tensor,
        video_scratch_host: torch.Tensor,
        audio_hidden_host: torch.Tensor,
        audio_scratch_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
        blocks: Iterable[tuple[LTX2RecomputeProjections, LTX2BlockOps]],
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        stats: LTX2DiTStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = LTX2DiTStats() if stats is None else stats
        video_source, video_destination = video_hidden_host, video_scratch_host
        audio_source, audio_destination = audio_hidden_host, audio_scratch_host
        for projections, ops in blocks:
            self.run_block(
                video_source,
                video_destination,
                audio_source,
                audio_destination,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                self_softmax_scale=self_softmax_scale,
                cross_softmax_scale=cross_softmax_scale,
                stats=stats,
            )
            video_source, video_destination = video_destination, video_source
            audio_source, audio_destination = audio_destination, audio_source
        return video_source, audio_source


__all__ = ["LTX2RecomputeRunner"]
