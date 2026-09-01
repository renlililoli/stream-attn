from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import RecomputedAttentionRunner, RecomputedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    TiledHostStageRunner,
    require_distinct_storage,
    validate_hidden_host,
)
from .stats import LTX2DiTStats
from .types import LTX2AttentionOps, LTX2BlockOps, LTX2RecomputeProjections, LTX2SequenceMeta


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
        self._validate_plans()
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
        if any(plan.output_mode != "device_consumer" for plan in plans):
            raise ValueError("LTX2 recompute runners require device_consumer output mode")
        if len({plan.device for plan in plans}) != 1 or len({plan.dtype for plan in plans}) != 1:
            raise ValueError("LTX2 attention runners must use one device and dtype")
        video_tokens = self.video_self_attention.plan.max_q_tokens
        audio_tokens = self.audio_self_attention.plan.max_q_tokens
        if self.video_self_attention.plan.max_kv_tokens != video_tokens:
            raise ValueError("LTX2 video self-attention requires equal Q and K/V lengths")
        if self.audio_self_attention.plan.max_kv_tokens != audio_tokens:
            raise ValueError("LTX2 audio self-attention requires equal Q and K/V lengths")
        expected_q = (
            (self.video_text_attention, video_tokens),
            (self.video_from_audio_attention, video_tokens),
            (self.audio_text_attention, audio_tokens),
            (self.audio_from_video_attention, audio_tokens),
        )
        if any(runner.plan.max_q_tokens != tokens for runner, tokens in expected_q):
            raise ValueError("LTX2 cross-attention Q lengths must match their target streams")
        if self.video_from_audio_attention.plan.max_kv_tokens != audio_tokens:
            raise ValueError("video-from-audio K/V length must match the audio stream")
        if self.audio_from_video_attention.plan.max_kv_tokens != video_tokens:
            raise ValueError("audio-from-video K/V length must match the video stream")

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

    @staticmethod
    def _run_self(
        runner: RecomputedAttentionRunner,
        source: torch.Tensor,
        destination: torch.Tensor,
        cu_seqlens: torch.Tensor,
        projection,
        ops: LTX2AttentionOps,
        consumer: AttentionOutputConsumer,
        stats,
        softmax_scale: float | None,
    ) -> None:
        consumer.reset(
            destination_hidden_host=destination,
            residual_hidden_host=source,
            epilogue=ops.epilogue,
        )
        with projection.context(), ops.context():
            runner.run_with_device_consumer(
                source,
                cu_seqlens,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                output_consumer=consumer,
                softmax_scale=softmax_scale,
                stats=stats,
            )

    @staticmethod
    def _run_cross(
        runner: RecomputedCrossAttentionRunner,
        query: torch.Tensor,
        context: torch.Tensor,
        destination: torch.Tensor,
        cu_q: torch.Tensor,
        cu_kv: torch.Tensor,
        projection,
        ops: LTX2AttentionOps,
        consumer: AttentionOutputConsumer,
        stats,
        softmax_scale: float | None,
    ) -> None:
        consumer.reset(
            destination_hidden_host=destination,
            residual_hidden_host=query,
            epilogue=ops.epilogue,
        )
        with projection.context(), ops.context():
            runner.run_with_device_consumer(
                query,
                context,
                cu_q,
                cu_kv,
                project_q=projection.project_q,
                project_kv=projection.project_kv,
                output_consumer=consumer,
                softmax_scale=softmax_scale,
                stats=stats,
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

        self._run_self(
            self.video_self_attention,
            video_source,
            video_destination,
            sequence_meta.video_cu_seqlens,
            projections.video_self_attention,
            ops.video_self_attention,
            self.video_consumer,
            stats.video_self_recompute,
            self_softmax_scale,
        )
        self._run_self(
            self.audio_self_attention,
            audio_source,
            audio_destination,
            sequence_meta.audio_cu_seqlens,
            projections.audio_self_attention,
            ops.audio_self_attention,
            self.audio_consumer,
            stats.audio_self_recompute,
            self_softmax_scale,
        )
        self._run_cross(
            self.video_text_attention,
            video_destination,
            text_hidden_host,
            video_source,
            sequence_meta.video_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.video_text_attention,
            ops.video_text_attention,
            self.video_consumer,
            stats.video_text_recompute,
            cross_softmax_scale,
        )
        self._run_cross(
            self.audio_text_attention,
            audio_destination,
            text_hidden_host,
            audio_source,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.text_cu_seqlens,
            projections.audio_text_attention,
            ops.audio_text_attention,
            self.audio_consumer,
            stats.audio_text_recompute,
            cross_softmax_scale,
        )

        # Both directions read the same post-text snapshots and write alternate buffers.
        self._run_cross(
            self.video_from_audio_attention,
            video_source,
            audio_source,
            video_destination,
            sequence_meta.video_cu_seqlens,
            sequence_meta.audio_cu_seqlens,
            projections.video_from_audio_attention,
            ops.video_from_audio_attention,
            self.video_consumer,
            stats.video_from_audio_recompute,
            cross_softmax_scale,
        )
        self._run_cross(
            self.audio_from_video_attention,
            audio_source,
            video_source,
            audio_destination,
            sequence_meta.audio_cu_seqlens,
            sequence_meta.video_cu_seqlens,
            projections.audio_from_video_attention,
            ops.audio_from_video_attention,
            self.audio_consumer,
            stats.audio_from_video_recompute,
            cross_softmax_scale,
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = kwargs.pop("stats", None)
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
                stats=stats,
                **kwargs,
            )
            video_source, video_destination = video_destination, video_source
            audio_source, audio_destination = audio_destination, audio_source
        return video_source, audio_source


__all__ = ["LTX2RecomputeRunner"]
