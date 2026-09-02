from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise

import torch

from ...projection.contracts import KVTileProjector, QKVProjector, QTileProjector
from ...validation import validate_cu_seqlens
from ..common import AttentionEpilogue, DeviceTileOp, LeaseFactory


@dataclass(frozen=True)
class H3MaterializedPlan:
    projection_tile_tokens: int
    q_chunk_tokens: int
    kv_chunk_tokens: int
    ffn_tile_tokens: int
    estimated_workspace_bytes: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class H3RecomputePlan:
    q_chunk_tokens: int
    kv_chunk_tokens: int
    ffn_tile_tokens: int
    hidden_staging_tokens: int
    estimated_workspace_bytes: int

    def validate(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class H3MaterializedProjection:
    project_qkv: QKVProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class H3RecomputeProjection:
    project_q: QTileProjector
    project_kv: KVTileProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class H3BlockOps:
    attention_epilogue: AttentionEpilogue
    ffn: DeviceTileOp
    consumer_lease: LeaseFactory | None = None

    def consumer_context(self) -> AbstractContextManager:
        return nullcontext() if self.consumer_lease is None else self.consumer_lease()


@dataclass(frozen=True)
class H3SequenceMeta:
    cu_seqlens: torch.Tensor
    position_ids_gpu: torch.Tensor | None = None
    modulation_row_ids_gpu: torch.Tensor | None = None
    exact_prefix_tokens: tuple[int, ...] | None = None

    def validate(self, tokens: int) -> None:
        bounds = validate_cu_seqlens(
            self.cu_seqlens,
            tokens,
            "cu_seqlens",
            expected_dtype=torch.int32,
        )
        if self.exact_prefix_tokens is None:
            return
        if len(self.exact_prefix_tokens) != len(bounds) - 1:
            raise ValueError("exact_prefix_tokens must contain one value per packed segment")
        for index, (prefix, (start, stop)) in enumerate(
            zip(self.exact_prefix_tokens, pairwise(bounds))
        ):
            if isinstance(prefix, bool) or not isinstance(prefix, int):
                raise TypeError(f"exact_prefix_tokens[{index}] must be an integer")
            if not 0 <= prefix <= stop - start:
                raise ValueError(f"exact_prefix_tokens[{index}] exceeds its packed segment")


@dataclass(frozen=True)
class H3DenoisingStep:
    step_index: int
    total_steps: int

    def validate(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("step_index must be an integer")
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int):
            raise TypeError("total_steps must be an integer")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.step_index < self.total_steps:
            raise ValueError("step_index must lie within [0, total_steps)")


def _validate_aux_workspace_args(
    *,
    hidden_features: int,
    ffn_tile_tokens: int,
    num_final_output_buffers: int,
) -> None:
    for name, value in (
        ("hidden_features", hidden_features),
        ("ffn_tile_tokens", ffn_tile_tokens),
        ("num_final_output_buffers", num_final_output_buffers),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def estimate_h3_consumer_workspace_bytes(
    *,
    hidden_features: int,
    dtype: torch.dtype,
    ffn_tile_tokens: int,
    num_final_output_buffers: int = 2,
    final_output_chunk_tokens: int | None = None,
) -> int:
    _validate_aux_workspace_args(
        hidden_features=hidden_features,
        ffn_tile_tokens=ffn_tile_tokens,
        num_final_output_buffers=num_final_output_buffers,
    )
    final_output_chunk_tokens = (
        ffn_tile_tokens if final_output_chunk_tokens is None else final_output_chunk_tokens
    )
    if final_output_chunk_tokens <= 0:
        raise ValueError("final_output_chunk_tokens must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        (ffn_tile_tokens + num_final_output_buffers * final_output_chunk_tokens)
        * hidden_features
        * element_size
    )


def estimate_h3_materialized_aux_workspace_bytes(
    *,
    hidden_features: int,
    dtype: torch.dtype,
    projection_tile_tokens: int,
    num_projection_buffers: int,
    ffn_tile_tokens: int,
    num_final_output_buffers: int = 2,
) -> int:
    for name, value in (
        ("projection_tile_tokens", projection_tile_tokens),
        ("num_projection_buffers", num_projection_buffers),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    projection = num_projection_buffers * projection_tile_tokens * hidden_features * element_size
    return projection + estimate_h3_consumer_workspace_bytes(
        hidden_features=hidden_features,
        dtype=dtype,
        ffn_tile_tokens=ffn_tile_tokens,
        num_final_output_buffers=num_final_output_buffers,
    )


def estimate_h3_recompute_aux_workspace_bytes(
    *,
    hidden_features: int,
    dtype: torch.dtype,
    hidden_staging_tokens: int,
    ffn_tile_tokens: int,
    num_final_output_buffers: int = 2,
) -> int:
    if hidden_staging_tokens <= 0:
        raise ValueError("hidden_staging_tokens must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    staging = hidden_staging_tokens * hidden_features * element_size
    return staging + estimate_h3_consumer_workspace_bytes(
        hidden_features=hidden_features,
        dtype=dtype,
        ffn_tile_tokens=ffn_tile_tokens,
        num_final_output_buffers=num_final_output_buffers,
    )


__all__ = [
    "AttentionEpilogue",
    "DeviceTileOp",
    "H3BlockOps",
    "H3DenoisingStep",
    "H3MaterializedPlan",
    "H3MaterializedProjection",
    "H3RecomputePlan",
    "H3RecomputeProjection",
    "H3SequenceMeta",
    "LeaseFactory",
    "estimate_h3_consumer_workspace_bytes",
    "estimate_h3_materialized_aux_workspace_bytes",
    "estimate_h3_recompute_aux_workspace_bytes",
]
