from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import ProjectedAttentionRunner, ProjectedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    TiledHostStageRunner,
    validate_hidden_host,
)
from .stats import LTX2DiTStats
from .types import LTX2AttentionOps, LTX2BlockOps, LTX2MaterializedProjections, LTX2SequenceMeta


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
        self._validate_plans()
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

    def _validate_plans(self) -> None:
        runners = (
            self.video_self_attention,
            self.audio_self_attention,
            self.video_text_attention,
            self.audio_text_attention,
            self.video_from_audio_attention,
            self.audio_from_video_attention,
        )
        plans = tuple(runner.plan for runner in runners)
        if self.video_hidden_features <= 0 or self.audio_hidden_features <= 0:
            raise ValueError("LTX2 hidden feature sizes must be positive")
        if any(plan.output_mode != "device_consumer" for plan in plans):
            raise ValueError("LTX2 attention runners require device_consumer output mode")
        if len({plan.device for plan in plans}) != 1 or len({plan.dtype for plan in plans}) != 1:
            raise ValueError("LTX2 attention runners must use one device and dtype")
        video_tokens = self.video_self_attention.plan.max_q_tokens
        audio_tokens = self.audio_self_attention.plan.max_q_tokens
        if self.video_self_attention.plan.max_kv_tokens != video_tokens:
            raise ValueError("LTX2 video self-attention must plan equal Q and K/V lengths")
        if self.audio_self_attention.plan.max_kv_tokens != audio_tokens:
            raise ValueError("LTX2 audio self-attention must plan equal Q and K/V lengths")
        if any(
            runner.plan.max_q_tokens != expected
            for runner, expected in (
                (self.video_text_attention, video_tokens),
                (self.video_from_audio_attention, video_tokens),
                (self.audio_text_attention, audio_tokens),
                (self.audio_from_video_attention, audio_tokens),
            )
        ):
            raise ValueError("LTX2 cross-attention Q lengths must match their target streams")
        if self.video_from_audio_attention.plan.max_kv_tokens != audio_tokens:
            raise ValueError("video-from-audio K/V length must match the audio stream")
        if self.audio_from_video_attention.plan.max_kv_tokens != video_tokens:
            raise ValueError("audio-from-video K/V length must match the video stream")
        if (
            self.video_text_attention.plan.max_kv_tokens
            != self.audio_text_attention.plan.max_kv_tokens
        ):
            raise ValueError("LTX2 text cross-attention runners must plan the same text length")

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

    def _run_self_attention(
        self,
        runner: ProjectedAttentionRunner,
        hidden_host: torch.Tensor,
        cu_seqlens: torch.Tensor,
        projection,
        ops: LTX2AttentionOps,
        consumer: AttentionOutputConsumer,
        stats,
        softmax_scale: float | None,
    ) -> None:
        started = time.perf_counter()
        with projection.context():
            q, k, v = runner.project_qkv_to_host(
                hidden_host,
                projection.project_qkv,
                stats=stats,
            )
        raw_output_bytes = q.numel() * q.element_size()
        stats.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        consumer.reset(
            destination_hidden_host=hidden_host,
            residual_hidden_host=hidden_host,
            epilogue=ops.epilogue,
        )
        attention_started = time.perf_counter()
        with ops.context():
            runner.attention.run_with_device_consumer(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                output_consumer=consumer,
                softmax_scale=softmax_scale,
                stats=stats.attention,
            )
        stats.attention_output_seconds += time.perf_counter() - attention_started
        stats.wall_seconds += time.perf_counter() - started

    def _run_cross_attention(
        self,
        runner: ProjectedCrossAttentionRunner,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        projection,
        ops: LTX2AttentionOps,
        consumer: AttentionOutputConsumer,
        stats,
        softmax_scale: float | None,
    ) -> None:
        started = time.perf_counter()
        with projection.context():
            q, k, v = runner.project_to_host(
                query_hidden_host,
                context_hidden_host,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                stats=stats,
            )
        raw_output_bytes = q.numel() * q.element_size()
        stats.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        consumer.reset(
            destination_hidden_host=query_hidden_host,
            residual_hidden_host=query_hidden_host,
            epilogue=ops.epilogue,
        )
        attention_started = time.perf_counter()
        with ops.context():
            runner.attention.run_with_device_consumer(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                output_consumer=consumer,
                softmax_scale=softmax_scale,
                stats=stats.attention,
            )
        stats.attention_output_seconds += time.perf_counter() - attention_started
        stats.wall_seconds += time.perf_counter() - started

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

        self._run_self_attention(
            self.video_self_attention,
            video_hidden_host,
            sequence_meta.video_cu_seqlens,
            projections.video_self_attention,
            ops.video_self_attention,
            self.video_consumer,
            stats.video_self_attention,
            self_softmax_scale,
        )
        self._run_self_attention(
            self.audio_self_attention,
            audio_hidden_host,
            sequence_meta.audio_cu_seqlens,
            projections.audio_self_attention,
            ops.audio_self_attention,
            self.audio_consumer,
            stats.audio_self_attention,
            self_softmax_scale,
        )
        self._run_cross_attention(
            self.video_text_attention,
            video_hidden_host,
            text_hidden_host,
            sequence_meta.video_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.video_text_attention,
            ops.video_text_attention,
            self.video_consumer,
            stats.video_text_attention,
            cross_softmax_scale,
        )
        self._run_cross_attention(
            self.audio_text_attention,
            audio_hidden_host,
            text_hidden_host,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.audio_text_attention,
            ops.audio_text_attention,
            self.audio_consumer,
            stats.audio_text_attention,
            cross_softmax_scale,
        )

        # Materialize both directions from the same pre-cross host snapshots.
        video_projection_before = stats.video_from_audio_attention.projection_seconds
        with projections.video_from_audio_attention.context():
            video_q, audio_k, audio_v = self.video_from_audio_attention.project_to_host(
                video_hidden_host,
                audio_hidden_host,
                project_q=projections.video_from_audio_attention.project_q,
                project_kv=projections.video_from_audio_attention.project_kv,
                stats=stats.video_from_audio_attention,
            )
        video_projection_seconds = (
            stats.video_from_audio_attention.projection_seconds - video_projection_before
        )
        audio_projection_before = stats.audio_from_video_attention.projection_seconds
        with projections.audio_from_video_attention.context():
            audio_q, video_k, video_v = self.audio_from_video_attention.project_to_host(
                audio_hidden_host,
                video_hidden_host,
                project_q=projections.audio_from_video_attention.project_q,
                project_kv=projections.audio_from_video_attention.project_kv,
                stats=stats.audio_from_video_attention,
            )

        audio_projection_seconds = (
            stats.audio_from_video_attention.projection_seconds - audio_projection_before
        )
        video_raw_output_bytes = video_q.numel() * video_q.element_size()
        audio_raw_output_bytes = audio_q.numel() * audio_q.element_size()
        stats.video_from_audio_attention.raw_attention_roundtrip_bytes_avoided += (
            2 * video_raw_output_bytes
        )
        stats.audio_from_video_attention.raw_attention_roundtrip_bytes_avoided += (
            2 * audio_raw_output_bytes
        )

        self.video_consumer.reset(
            destination_hidden_host=video_hidden_host,
            residual_hidden_host=video_hidden_host,
            epilogue=ops.video_from_audio_attention.epilogue,
        )
        video_attention_started = time.perf_counter()
        with ops.video_from_audio_attention.context():
            self.video_from_audio_attention.attention.run_with_device_consumer(
                video_q,
                audio_k,
                audio_v,
                sequence_meta.video_cu_seqlens,
                sequence_meta.audio_cu_seqlens,
                output_consumer=self.video_consumer,
                softmax_scale=cross_softmax_scale,
                stats=stats.video_from_audio_attention.attention,
            )
        video_attention_seconds = time.perf_counter() - video_attention_started
        stats.video_from_audio_attention.attention_output_seconds += video_attention_seconds
        stats.video_from_audio_attention.wall_seconds += (
            video_projection_seconds + video_attention_seconds
        )
        self.audio_consumer.reset(
            destination_hidden_host=audio_hidden_host,
            residual_hidden_host=audio_hidden_host,
            epilogue=ops.audio_from_video_attention.epilogue,
        )
        audio_attention_started = time.perf_counter()
        with ops.audio_from_video_attention.context():
            self.audio_from_video_attention.attention.run_with_device_consumer(
                audio_q,
                video_k,
                video_v,
                sequence_meta.audio_cu_seqlens,
                sequence_meta.video_cu_seqlens,
                output_consumer=self.audio_consumer,
                softmax_scale=cross_softmax_scale,
                stats=stats.audio_from_video_attention.attention,
            )
        audio_attention_seconds = time.perf_counter() - audio_attention_started
        stats.audio_from_video_attention.attention_output_seconds += audio_attention_seconds
        stats.audio_from_video_attention.wall_seconds += (
            audio_projection_seconds + audio_attention_seconds
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = kwargs.pop("stats", None)
        stats = LTX2DiTStats() if stats is None else stats
        for projections, ops in blocks:
            self.run_block_(
                video_hidden_host,
                audio_hidden_host,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                stats=stats,
                **kwargs,
            )
        return video_hidden_host, audio_hidden_host


__all__ = ["LTX2MaterializedRunner"]
