from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise

import torch

from ...projection import (
    CrossProjection,
    CrossRecomputeProjection,
    SelfProjection,
    SelfRecomputeProjection,
)
from ..common import AttentionEpilogue, DeviceTileOp, LeaseFactory


@dataclass(frozen=True)
class WanSequenceMeta:
    hidden_cu_seqlens: torch.Tensor
    text_cu_seqlens: torch.Tensor

    @staticmethod
    def _validate_bounds(name: str, bounds: torch.Tensor, tokens: int) -> None:
        if bounds.device.type != "cpu" or bounds.dtype != torch.int32 or bounds.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional CPU int32 tensor")
        if bounds.numel() < 2 or int(bounds[0]) != 0 or int(bounds[-1]) != tokens:
            raise ValueError(f"{name} must span the complete token tensor")
        values = bounds.to(dtype=torch.int64).tolist()
        if any(stop < start for start, stop in pairwise(values)):
            raise ValueError(f"{name} must be non-decreasing")

    def validate(self, hidden_tokens: int, text_tokens: int) -> None:
        self._validate_bounds("hidden_cu_seqlens", self.hidden_cu_seqlens, hidden_tokens)
        self._validate_bounds("text_cu_seqlens", self.text_cu_seqlens, text_tokens)
        if self.hidden_cu_seqlens.numel() != self.text_cu_seqlens.numel():
            raise ValueError("hidden and text packed boundaries must describe the same batch")


@dataclass(frozen=True)
class WanMaterializedProjections:
    self_attention: SelfProjection
    text_cross_attention: CrossProjection


@dataclass(frozen=True)
class WanRecomputeProjections:
    self_attention: SelfRecomputeProjection
    text_cross_attention: CrossRecomputeProjection


@dataclass(frozen=True)
class WanBlockOps:
    self_attention_epilogue: AttentionEpilogue
    cross_attention_epilogue: AttentionEpilogue
    ffn: DeviceTileOp
    self_attention_lease: LeaseFactory | None = None
    cross_attention_lease: LeaseFactory | None = None
    ffn_lease: LeaseFactory | None = None

    @staticmethod
    def _context(lease: LeaseFactory | None) -> AbstractContextManager:
        return nullcontext() if lease is None else lease()

    def self_attention_context(self) -> AbstractContextManager:
        return self._context(self.self_attention_lease)

    def cross_attention_context(self) -> AbstractContextManager:
        return self._context(self.cross_attention_lease)

    def ffn_context(self) -> AbstractContextManager:
        return self._context(self.ffn_lease)


__all__ = [
    "WanBlockOps",
    "WanMaterializedProjections",
    "WanRecomputeProjections",
    "WanSequenceMeta",
]
