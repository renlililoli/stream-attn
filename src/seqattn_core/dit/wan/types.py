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
from ..common import AttentionEpilogue, DeviceTileOp, LeaseFactory


@dataclass(frozen=True)
class WanSequenceMeta:
    hidden_cu_seqlens: torch.Tensor
    text_cu_seqlens: torch.Tensor

    @staticmethod
    def _validate_bounds(name: str, bounds: torch.Tensor, tokens: int) -> None:
        validate_cu_seqlens(
            bounds,
            tokens,
            name,
            expected_dtype=torch.int32,
        )

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


def validate_wan_runner_contract(
    self_attention: ProjectedAttentionRunner | RecomputedAttentionRunner,
    cross_attention: ProjectedCrossAttentionRunner | RecomputedCrossAttentionRunner,
    *,
    hidden_features: int,
) -> None:
    if hidden_features <= 0:
        raise ValueError("hidden_features must be positive")
    self_plan = self_attention.plan
    cross_plan = cross_attention.plan
    if self_plan.output_mode != "device_consumer" or cross_plan.output_mode != "device_consumer":
        raise ValueError("Wan attention runners require device_consumer output mode")
    if self_plan.device != cross_plan.device or self_plan.dtype != cross_plan.dtype:
        raise ValueError("Wan self and cross attention must use the same device and dtype")
    if self_plan.max_q_tokens != cross_plan.max_q_tokens:
        raise ValueError("Wan self and cross attention must plan the same hidden token count")
    if self_plan.max_q_tokens != self_plan.max_kv_tokens:
        raise ValueError("Wan self-attention requires equal planned Q and K/V token counts")
    if cross_attention.query_hidden_features != hidden_features:
        raise ValueError("Wan cross-attention query feature size must match hidden features")


__all__ = [
    "WanBlockOps",
    "WanMaterializedProjections",
    "WanRecomputeProjections",
    "WanSequenceMeta",
    "validate_wan_runner_contract",
]
