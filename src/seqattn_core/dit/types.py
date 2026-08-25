from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

from ..projection.types import QKVProjector

DeviceTileOp = Callable[[torch.Tensor, int, int], torch.Tensor]
LeaseFactory = Callable[[], AbstractContextManager]


@dataclass(frozen=True)
class H3ChunkPlan:
    projection_chunk_tokens: int
    q_chunk_tokens: int
    kv_chunk_tokens: int
    mlp_chunk_tokens: int
    estimated_workspace_bytes: int

    def validate(self) -> None:
        for name, value in (
            ("projection_chunk_tokens", self.projection_chunk_tokens),
            ("q_chunk_tokens", self.q_chunk_tokens),
            ("kv_chunk_tokens", self.kv_chunk_tokens),
            ("mlp_chunk_tokens", self.mlp_chunk_tokens),
            ("estimated_workspace_bytes", self.estimated_workspace_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class H3BlockOps:
    project_qkv: QKVProjector
    attention_epilogue: DeviceTileOp
    mlp: DeviceTileOp
    qkv_lease: LeaseFactory | None = None
    consumer_lease: LeaseFactory | None = None

    def qkv_context(self) -> AbstractContextManager:
        return nullcontext() if self.qkv_lease is None else self.qkv_lease()

    def consumer_context(self) -> AbstractContextManager:
        return nullcontext() if self.consumer_lease is None else self.consumer_lease()


@dataclass(frozen=True)
class H3SequenceMeta:
    cu_seqlens: torch.Tensor
    position_ids_gpu: torch.Tensor | None = None
    modulation_row_ids_gpu: torch.Tensor | None = None

    def validate(self, tokens: int) -> None:
        if self.cu_seqlens.device.type != "cpu":
            raise ValueError("cu_seqlens must be CPU-resident")
        if self.cu_seqlens.dtype != torch.int32 or self.cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
        if self.cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least two boundaries")
        if int(self.cu_seqlens[0]) != 0 or int(self.cu_seqlens[-1]) != tokens:
            raise ValueError("cu_seqlens must span the complete hidden tensor")


def estimate_h3_aux_workspace_bytes(
    *,
    hidden_features: int,
    dtype: torch.dtype,
    projection_chunk_tokens: int,
    num_projection_buffers: int,
    mlp_chunk_tokens: int,
    num_final_output_buffers: int = 2,
) -> int:
    if hidden_features <= 0:
        raise ValueError("hidden_features must be positive")
    for name, value in (
        ("projection_chunk_tokens", projection_chunk_tokens),
        ("num_projection_buffers", num_projection_buffers),
        ("mlp_chunk_tokens", mlp_chunk_tokens),
        ("num_final_output_buffers", num_final_output_buffers),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    projection = (
        num_projection_buffers * projection_chunk_tokens * hidden_features * element_size
    )
    consumer = (
        (1 + num_final_output_buffers) * mlp_chunk_tokens * hidden_features * element_size
    )
    return projection + consumer


__all__ = [
    "DeviceTileOp",
    "H3BlockOps",
    "H3ChunkPlan",
    "H3SequenceMeta",
    "LeaseFactory",
    "estimate_h3_aux_workspace_bytes",
]
