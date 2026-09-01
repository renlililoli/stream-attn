from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise

import torch

from ...projection import CrossProjection, SelfProjection
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
        if bounds.device.type != "cpu" or bounds.dtype != torch.int32 or bounds.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional CPU int32 tensor")
        if bounds.numel() < 2 or int(bounds[0]) != 0 or int(bounds[-1]) != tokens:
            raise ValueError(f"{name} must span the complete token tensor")
        values = bounds.to(dtype=torch.int64).tolist()
        if any(stop < start for start, stop in pairwise(values)):
            raise ValueError(f"{name} must be non-decreasing")

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


__all__ = [
    "LTX2AttentionOps",
    "LTX2BlockOps",
    "LTX2MaterializedProjections",
    "LTX2SequenceMeta",
]
