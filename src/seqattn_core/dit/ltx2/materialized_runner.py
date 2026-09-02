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
from .stats import LTX2DiTStats
from .types import (
    LTX2BlockOps,
    LTX2MaterializedProjections,
    LTX2SequenceMeta,
    validate_ltx2_runner_contract,
)


class LTX2MaterializedRunner:
    """Fixed-order LTX2 video/audio block with snapshot-safe bidirectional cross-attention."""

    def __init__(
        self,
        *,
        video_self_attention: ProjectedAttentionRunner,
        audio_self_attention: ProjectedAttentionRunner,
        video_text_attention: ProjectedCrossAttentionRunner,
        audio_text_attention: ProjectedCrossAttentionRunner,
        video_from_audio_attention: ProjectedCrossAttentionRunner,
        audio_from_video_attention: ProjectedCrossAttentionRunner,
        video_hidden_features: int,
        audio_hidden_features: int,
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
        self.video_hidden_features = video_hidden_features
        self.audio_hidden_features = audio_hidden_features
        validate_ltx2_runner_contract(
            video_self_attention=video_self_attention,
            audio_self_attention=audio_self_attention,
            video_text_attention=video_text_attention,
            audio_text_attention=audio_text_attention,
            video_from_audio_attention=video_from_audio_attention,
            audio_from_video_attention=audio_from_video_attention,
            video_hidden_features=video_hidden_features,
            audio_hidden_features=audio_hidden_features,
        )
        video_plan = video_self_attention.plan
        audio_plan = audio_self_attention.plan
        self.video_consumer = AttentionOutputConsumer(
            AttentionOutputWorkspace(
                hidden_features=video_hidden_features,
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
                hidden_features=audio_hidden_features,
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
        self.video_attention_executor = MaterializedAttentionExecutor(self.video_consumer)
        self.audio_attention_executor = MaterializedAttentionExecutor(self.audio_consumer)
        require_pinned = video_self_attention.pipeline_config.require_pinned_hidden
        self.video_ffn = TiledHostStageRunner(
            hidden_features=video_hidden_features,
            chunk_tokens=video_ffn_tile_tokens,
            dtype=video_plan.dtype,
            device=video_plan.device,
            require_pinned_hidden=require_pinned,
        )
        self.audio_ffn = TiledHostStageRunner(
            hidden_features=audio_hidden_features,
            chunk_tokens=audio_ffn_tile_tokens,
            dtype=audio_plan.dtype,
            device=audio_plan.device,
            require_pinned_hidden=require_pinned,
        )

    def _validate_inputs(
        self,
        video_hidden_host: torch.Tensor,
        audio_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
    ) -> None:
        require_pinned = self.video_self_attention.pipeline_config.require_pinned_hidden
        validate_hidden_host(
            video_hidden_host,
            plan=self.video_self_attention.plan,
            hidden_features=self.video_hidden_features,
            require_pinned=require_pinned,
            name="video_hidden_host",
        )
        validate_hidden_host(
            audio_hidden_host,
            plan=self.audio_self_attention.plan,
            hidden_features=self.audio_hidden_features,
            require_pinned=require_pinned,
            name="audio_hidden_host",
        )
        self.video_text_attention.validate_inputs(video_hidden_host, text_hidden_host)
        self.audio_text_attention.validate_inputs(audio_hidden_host, text_hidden_host)
        sequence_meta.validate(
            video_hidden_host.shape[0],
            audio_hidden_host.shape[0],
            text_hidden_host.shape[0],
        )

    @single_flight
    @torch.inference_mode()
    def run_block_(
        self,
        video_hidden_host: torch.Tensor,
        audio_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
        projections: LTX2MaterializedProjections,
        ops: LTX2BlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        stats: LTX2DiTStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            video_hidden_host,
            audio_hidden_host,
            text_hidden_host,
            sequence_meta,
        )
        stats = LTX2DiTStats() if stats is None else stats
        stats.backend = self.video_self_attention.attention.backend
        stats.qkv_storage_policy = "materialized"
        started = time.perf_counter()

        self.video_attention_executor.run_self(
            self.video_self_attention,
            video_hidden_host,
            sequence_meta.video_cu_seqlens,
            projections.video_self_attention,
            epilogue=ops.video_self_attention.epilogue,
            consumer_lease=ops.video_self_attention.weight_lease,
            softmax_scale=self_softmax_scale,
            causal=False,
            stats=stats.video_self_attention,
        )
        self.audio_attention_executor.run_self(
            self.audio_self_attention,
            audio_hidden_host,
            sequence_meta.audio_cu_seqlens,
            projections.audio_self_attention,
            epilogue=ops.audio_self_attention.epilogue,
            consumer_lease=ops.audio_self_attention.weight_lease,
            softmax_scale=self_softmax_scale,
            causal=False,
            stats=stats.audio_self_attention,
        )
        self.video_attention_executor.run_cross(
            self.video_text_attention,
            video_hidden_host,
            text_hidden_host,
            sequence_meta.video_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.video_text_attention,
            epilogue=ops.video_text_attention.epilogue,
            consumer_lease=ops.video_text_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.video_text_attention,
        )
        self.audio_attention_executor.run_cross(
            self.audio_text_attention,
            audio_hidden_host,
            text_hidden_host,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.audio_text_attention,
            epilogue=ops.audio_text_attention.epilogue,
            consumer_lease=ops.audio_text_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            stats=stats.audio_text_attention,
        )

        # Materialize both directions from the same pre-cross host snapshots.
        video_from_audio = self.video_attention_executor.materialize_cross(
            self.video_from_audio_attention,
            video_hidden_host,
            audio_hidden_host,
            projections.video_from_audio_attention,
            stats.video_from_audio_attention,
        )
        audio_from_video = self.audio_attention_executor.materialize_cross(
            self.audio_from_video_attention,
            audio_hidden_host,
            video_hidden_host,
            projections.audio_from_video_attention,
            stats.audio_from_video_attention,
        )
        self.video_attention_executor.consume(
            self.video_from_audio_attention,
            video_from_audio,
            sequence_meta.video_cu_seqlens,
            sequence_meta.audio_cu_seqlens,
            destination_hidden_host=video_hidden_host,
            residual_hidden_host=video_hidden_host,
            epilogue=ops.video_from_audio_attention.epilogue,
            consumer_lease=ops.video_from_audio_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            causal=False,
            stats=stats.video_from_audio_attention,
        )
        self.audio_attention_executor.consume(
            self.audio_from_video_attention,
            audio_from_video,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.video_cu_seqlens,
            destination_hidden_host=audio_hidden_host,
            residual_hidden_host=audio_hidden_host,
            epilogue=ops.audio_from_video_attention.epilogue,
            consumer_lease=ops.audio_from_video_attention.weight_lease,
            softmax_scale=cross_softmax_scale,
            causal=False,
            stats=stats.audio_from_video_attention,
        )

        with ops.video_ffn.context():
            self.video_ffn.run(
                video_hidden_host,
                video_hidden_host,
                ops.video_ffn.operation,
                stats=stats.video_ffn,
            )
        with ops.audio_ffn.context():
            self.audio_ffn.run(
                audio_hidden_host,
                audio_hidden_host,
                ops.audio_ffn.operation,
                stats=stats.audio_ffn,
            )

        runners = (
            self.video_self_attention,
            self.audio_self_attention,
            self.video_text_attention,
            self.audio_text_attention,
            self.video_from_audio_attention,
            self.audio_from_video_attention,
        )
        arenas = {id(runner.arena): runner.arena for runner in runners}
        stats.qkv_host_bytes_peak = max(
            stats.qkv_host_bytes_peak,
            sum(arena.allocated_bytes for arena in arenas.values()),
        )
        stats.hidden_host_bytes_peak = max(
            stats.hidden_host_bytes_peak,
            (video_hidden_host.numel() + audio_hidden_host.numel() + text_hidden_host.numel())
            * video_hidden_host.element_size(),
        )
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return video_hidden_host, audio_hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        video_hidden_host: torch.Tensor,
        audio_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: LTX2SequenceMeta,
        blocks: Iterable[tuple[LTX2MaterializedProjections, LTX2BlockOps]],
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        stats: LTX2DiTStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = LTX2DiTStats() if stats is None else stats
        for projections, ops in blocks:
            self.run_block_(
                video_hidden_host,
                audio_hidden_host,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                self_softmax_scale=self_softmax_scale,
                cross_softmax_scale=cross_softmax_scale,
                stats=stats,
            )
        return video_hidden_host, audio_hidden_host


__all__ = ["LTX2MaterializedRunner"]
