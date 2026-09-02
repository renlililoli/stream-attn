from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

from ...projection import (
    CrossProjection,
    CrossRecomputeProjection,
    ProjectedAttentionRunner,
    ProjectedCrossAttentionRunner,
    RecomputedAttentionRunner,
    RecomputedCrossAttentionRunner,
    SelfProjection,
    SelfRecomputeProjection,
)
from ...validation import validate_cu_seqlens
from ..common import (
    AttentionEpilogue,
    LeaseFactory,
    TiledStageOp,
    cu_seqlens_from_padding_mask,
    reject_additive_attention_mask,
)


@dataclass(frozen=True)
class LTX2SequenceMeta:
    video_cu_seqlens: torch.Tensor
    audio_cu_seqlens: torch.Tensor
    text_cu_seqlens: torch.Tensor

    @classmethod
    def from_padding_masks(
        cls,
        video_padding_mask: torch.Tensor,
        audio_padding_mask: torch.Tensor,
        text_padding_mask: torch.Tensor,
    ) -> LTX2SequenceMeta:
        for name, mask in (
            ("video_padding_mask", video_padding_mask),
            ("audio_padding_mask", audio_padding_mask),
            ("text_padding_mask", text_padding_mask),
        ):
            reject_additive_attention_mask(mask, name=name)
        if not (
            video_padding_mask.shape[0] == audio_padding_mask.shape[0] == text_padding_mask.shape[0]
        ):
            raise ValueError("LTX2 padding masks must describe the same batch")
        return cls(
            cu_seqlens_from_padding_mask(video_padding_mask, name="video_padding_mask"),
            cu_seqlens_from_padding_mask(audio_padding_mask, name="audio_padding_mask"),
            cu_seqlens_from_padding_mask(text_padding_mask, name="text_padding_mask"),
        )

    @staticmethod
    def _validate_bounds(name: str, bounds: torch.Tensor, tokens: int) -> None:
        validate_cu_seqlens(
            bounds,
            tokens,
            name,
            expected_dtype=torch.int32,
        )

    def validate(self, video_tokens: int, audio_tokens: int, text_tokens: int) -> None:
        self._validate_bounds("video_cu_seqlens", self.video_cu_seqlens, video_tokens)
        self._validate_bounds("audio_cu_seqlens", self.audio_cu_seqlens, audio_tokens)
        self._validate_bounds("text_cu_seqlens", self.text_cu_seqlens, text_tokens)
        boundary_counts = {
            self.video_cu_seqlens.numel(),
            self.audio_cu_seqlens.numel(),
            self.text_cu_seqlens.numel(),
        }
        if len(boundary_counts) != 1:
            raise ValueError("LTX2 packed boundaries must describe the same batch")


@dataclass(frozen=True)
class LTX2MaterializedProjections:
    video_self_attention: SelfProjection
    audio_self_attention: SelfProjection
    video_text_attention: CrossProjection
    audio_text_attention: CrossProjection
    video_from_audio_attention: CrossProjection
    audio_from_video_attention: CrossProjection


@dataclass(frozen=True)
class LTX2RecomputeProjections:
    video_self_attention: SelfRecomputeProjection
    audio_self_attention: SelfRecomputeProjection
    video_text_attention: CrossRecomputeProjection
    audio_text_attention: CrossRecomputeProjection
    video_from_audio_attention: CrossRecomputeProjection
    audio_from_video_attention: CrossRecomputeProjection


@dataclass(frozen=True)
class LTX2AttentionOps:
    epilogue: AttentionEpilogue
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class LTX2BlockOps:
    video_self_attention: LTX2AttentionOps
    audio_self_attention: LTX2AttentionOps
    video_text_attention: LTX2AttentionOps
    audio_text_attention: LTX2AttentionOps
    video_from_audio_attention: LTX2AttentionOps
    audio_from_video_attention: LTX2AttentionOps
    video_ffn: TiledStageOp
    audio_ffn: TiledStageOp


def validate_ltx2_runner_contract(
    *,
    video_self_attention: ProjectedAttentionRunner | RecomputedAttentionRunner,
    audio_self_attention: ProjectedAttentionRunner | RecomputedAttentionRunner,
    video_text_attention: ProjectedCrossAttentionRunner | RecomputedCrossAttentionRunner,
    audio_text_attention: ProjectedCrossAttentionRunner | RecomputedCrossAttentionRunner,
    video_from_audio_attention: ProjectedCrossAttentionRunner | RecomputedCrossAttentionRunner,
    audio_from_video_attention: ProjectedCrossAttentionRunner | RecomputedCrossAttentionRunner,
    video_hidden_features: int,
    audio_hidden_features: int,
) -> None:
    if video_hidden_features <= 0 or audio_hidden_features <= 0:
        raise ValueError("LTX2 hidden feature sizes must be positive")
    runners = (
        video_self_attention,
        audio_self_attention,
        video_text_attention,
        audio_text_attention,
        video_from_audio_attention,
        audio_from_video_attention,
    )
    plans = tuple(runner.plan for runner in runners)
    if any(plan.output_mode != "device_consumer" for plan in plans):
        raise ValueError("LTX2 attention runners require device_consumer output mode")
    if len({plan.device for plan in plans}) != 1 or len({plan.dtype for plan in plans}) != 1:
        raise ValueError("LTX2 attention runners must use one device and dtype")

    video_tokens = video_self_attention.plan.max_q_tokens
    audio_tokens = audio_self_attention.plan.max_q_tokens
    if video_self_attention.plan.max_kv_tokens != video_tokens:
        raise ValueError("LTX2 video self-attention must plan equal Q and K/V lengths")
    if audio_self_attention.plan.max_kv_tokens != audio_tokens:
        raise ValueError("LTX2 audio self-attention must plan equal Q and K/V lengths")
    expected_q_tokens = (
        (video_text_attention, video_tokens),
        (video_from_audio_attention, video_tokens),
        (audio_text_attention, audio_tokens),
        (audio_from_video_attention, audio_tokens),
    )
    if any(runner.plan.max_q_tokens != tokens for runner, tokens in expected_q_tokens):
        raise ValueError("LTX2 cross-attention Q lengths must match their target streams")
    if video_from_audio_attention.plan.max_kv_tokens != audio_tokens:
        raise ValueError("video-from-audio K/V length must match the audio stream")
    if audio_from_video_attention.plan.max_kv_tokens != video_tokens:
        raise ValueError("audio-from-video K/V length must match the video stream")
    if video_text_attention.plan.max_kv_tokens != audio_text_attention.plan.max_kv_tokens:
        raise ValueError("LTX2 text cross-attention runners must plan the same text length")

    expected_features = (
        (video_text_attention, video_hidden_features, None),
        (audio_text_attention, audio_hidden_features, None),
        (video_from_audio_attention, video_hidden_features, audio_hidden_features),
        (audio_from_video_attention, audio_hidden_features, video_hidden_features),
    )
    for runner, query_features, context_features in expected_features:
        if runner.query_hidden_features != query_features:
            raise ValueError("LTX2 cross-attention query feature size does not match its stream")
        if context_features is not None and runner.context_hidden_features != context_features:
            raise ValueError("LTX2 cross-attention context feature size does not match its stream")
    if video_text_attention.context_hidden_features != audio_text_attention.context_hidden_features:
        raise ValueError("LTX2 text cross-attention runners must use one text feature size")


__all__ = [
    "LTX2AttentionOps",
    "LTX2BlockOps",
    "LTX2MaterializedProjections",
    "LTX2RecomputeProjections",
    "LTX2SequenceMeta",
    "validate_ltx2_runner_contract",
]
