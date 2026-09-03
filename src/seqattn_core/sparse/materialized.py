from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from ..kernels.sol_preprocess import SOL_BLOCK_TOKENS, encode_sol_kv
from ..stats import ProjectedAttentionStats, StreamingAttentionStats
from ..streaming.workspace import CudaWorkspace
from ..validation import validate_cu_seqlens
from .plan import SolStreamingPlan

if TYPE_CHECKING:
    from ..projection import ProjectedAttentionRunner
    from ..projection.contracts import QKVProjector


@dataclass(frozen=True)
class _SegmentLayout:
    token_start: int
    token_stop: int
    block_start: int
    block_stop: int

    @property
    def blocks(self) -> int:
        return self.block_stop - self.block_start


@dataclass(frozen=True)
class _ProjectionRange:
    token_start: int
    token_stop: int
    block_start: int
    block_stop: int


class SolMaterializedSource:
    """Pinned INT8 K/V and precomputed summaries for one materialized Sol call."""

    storage_dtype = "int8"

    def __init__(
        self,
        plan: SolStreamingPlan,
        dense: CudaWorkspace,
        *,
        k_storage: torch.Tensor,
        v_storage: torch.Tensor,
        pin_memory: bool,
    ) -> None:
        attention = plan.attention
        self.plan = plan
        self.dense = dense
        self.pin_memory = pin_memory
        self._shape = (
            attention.max_kv_tokens,
            attention.kv_heads,
            attention.head_dim,
        )
        self.q: torch.Tensor | None = None
        self.k_quantized = self._borrow_int8_storage(k_storage, "K")
        self.v_quantized = self._borrow_int8_storage(v_storage, "V")
        self.k_scales: torch.Tensor | None = None
        self.v_scales: torch.Tensor | None = None
        self.k_centroids: torch.Tensor | None = None
        self.value_sums: torch.Tensor | None = None
        self._block_capacity = 0
        self._segments: tuple[_SegmentLayout, ...] = ()
        self._ranges: dict[tuple[int, int], _ProjectionRange] = {}

    @property
    def allocated_bytes(self) -> int:
        tensors = (
            self.k_scales,
            self.v_scales,
            self.k_centroids,
            self.value_sums,
        )
        return sum(
            tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None
        )

    def _borrow_int8_storage(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"Sol materialized {name} storage must be contiguous CPU memory")
        if tensor.is_pinned() != self.pin_memory:
            raise ValueError(f"Sol materialized {name} storage pinning must match the pipeline")
        byte_storage = tensor.view(torch.int8).view(-1)
        required = math.prod(self._shape)
        if byte_storage.numel() < required:
            raise ValueError(f"Sol materialized {name} storage is too small for INT8 K/V")
        return byte_storage[:required].view(self._shape)

    @property
    def segments(self) -> tuple[_SegmentLayout, ...]:
        if not self._segments:
            raise RuntimeError("Sol materialized source has not been prepared")
        return self._segments

    def validate_layout(self, plan: SolStreamingPlan, bounds: list[int]) -> None:
        """Require execution metadata to match the layout used for encoding."""

        if plan != self.plan:
            raise ValueError("Sol materialized source and attention plans must match")
        expected = tuple((segment.token_start, segment.token_stop) for segment in self.segments)
        actual = tuple(pairwise(bounds))
        if actual != expected:
            raise ValueError("Sol materialized source segments must match the execution cu_seqlens")

    def _ensure_storage(self, total_blocks: int) -> None:
        attention = self.plan.attention
        if total_blocks <= self._block_capacity:
            return
        scale_shape = (total_blocks, attention.kv_heads)
        summary_shape = (total_blocks, attention.kv_heads, attention.head_dim)
        self.k_scales = torch.empty(
            scale_shape,
            dtype=torch.float16,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.v_scales = torch.empty_like(self.k_scales, pin_memory=self.pin_memory)
        self.k_centroids = torch.empty(
            summary_shape,
            dtype=attention.dtype,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.value_sums = torch.empty_like(
            self.k_centroids,
            pin_memory=self.pin_memory,
        )
        self._block_capacity = total_blocks

    def prepare(
        self,
        q: torch.Tensor,
        bounds: list[int],
        *,
        projection_tile_tokens: int,
    ) -> tuple[tuple[int, int], ...]:
        if q.device.type != "cpu" or q.dtype != self.plan.attention.dtype:
            raise ValueError("Sol materialized Q storage must be a matching CPU tensor")
        if projection_tile_tokens <= 0:
            raise ValueError("projection_tile_tokens must be positive")
        if (
            len(bounds) < 2
            or bounds[0] != 0
            or bounds[-1] <= 0
            or bounds[-1] > q.shape[0]
            or any(start > stop for start, stop in pairwise(bounds))
        ):
            raise ValueError("Sol materialized bounds must fit Q storage and be non-decreasing")

        segments: list[_SegmentLayout] = []
        total_blocks = 0
        for start, stop in pairwise(bounds):
            blocks = math.ceil((stop - start) / SOL_BLOCK_TOKENS)
            segments.append(_SegmentLayout(start, stop, total_blocks, total_blocks + blocks))
            total_blocks += blocks
        self._ensure_storage(total_blocks)
        self.q = q
        self._segments = tuple(segments)

        ranges: list[_ProjectionRange] = []
        aligned_tile = projection_tile_tokens - projection_tile_tokens % SOL_BLOCK_TOKENS
        for segment in segments:
            local_start = 0
            segment_tokens = segment.token_stop - segment.token_start
            while local_start < segment_tokens:
                remaining = segment_tokens - local_start
                if remaining > projection_tile_tokens:
                    if aligned_tile == 0:
                        raise ValueError(
                            "Sol encoded projection requires projection_tile_tokens >= 64"
                        )
                    tile_tokens = aligned_tile
                else:
                    tile_tokens = remaining
                local_stop = local_start + tile_tokens
                block_start = segment.block_start + local_start // SOL_BLOCK_TOKENS
                block_stop = block_start + math.ceil(tile_tokens / SOL_BLOCK_TOKENS)
                ranges.append(
                    _ProjectionRange(
                        segment.token_start + local_start,
                        segment.token_start + local_stop,
                        block_start,
                        block_stop,
                    )
                )
                local_start = local_stop
        self._ranges = {(item.token_start, item.token_stop): item for item in ranges}
        return tuple((item.token_start, item.token_stop) for item in ranges)

    def encode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start: int,
        stop: int,
    ) -> tuple[torch.Tensor, ...]:
        if (start, stop) not in self._ranges:
            raise ValueError("encoded projection range was not prepared")
        encoded = encode_sol_kv(k, v, kv_tokens=stop - start)
        return (q, *encoded)

    def copy_to_host(
        self,
        start: int,
        stop: int,
        payload: tuple[torch.Tensor, ...],
    ) -> None:
        if len(payload) != 7:
            raise ValueError("Sol encoded projection returned an invalid payload")
        q, k_quantized, v_quantized, k_scales, v_scales, k_centroids, value_sums = payload
        projection_range = self._ranges[(start, stop)]
        block_slice = slice(projection_range.block_start, projection_range.block_stop)
        assert self.q is not None
        assert self.k_scales is not None and self.v_scales is not None
        assert self.k_centroids is not None and self.value_sums is not None
        self.q[start:stop].copy_(q, non_blocking=self.q.is_pinned())
        self.k_quantized[start:stop].copy_(
            k_quantized[: stop - start],
            non_blocking=self.k_quantized.is_pinned(),
        )
        self.v_quantized[start:stop].copy_(
            v_quantized[: stop - start],
            non_blocking=self.v_quantized.is_pinned(),
        )
        self.k_scales[block_slice].copy_(k_scales, non_blocking=self.k_scales.is_pinned())
        self.v_scales[block_slice].copy_(v_scales, non_blocking=self.v_scales.is_pinned())
        self.k_centroids[block_slice].copy_(
            k_centroids,
            non_blocking=self.k_centroids.is_pinned(),
        )
        self.value_sums[block_slice].copy_(
            value_sums,
            non_blocking=self.value_sums.is_pinned(),
        )

    def load_q(
        self,
        destination: torch.Tensor,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None:
        assert self.q is not None
        workspace = self.dense
        with torch.cuda.stream(workspace.h2d_stream):
            if workspace.q_has_pending_compute:
                workspace.h2d_stream.wait_event(workspace.q_free)
            destination.copy_(self.q[start:stop], non_blocking=self.q.is_pinned())
            workspace.q_ready.record(workspace.h2d_stream)
        compute_stream.wait_event(workspace.q_ready)
        stats.h2d_bytes += (stop - start) * self.q.shape[1] * self.q.shape[2] * 2

    def load_summary(
        self,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        segment_id: int,
        stats: StreamingAttentionStats,
    ) -> int:
        segment = self.segments[segment_id]
        block_slice = slice(segment.block_start, segment.block_stop)
        assert self.k_centroids is not None and self.value_sums is not None
        destination_k[: segment.blocks].copy_(
            self.k_centroids[block_slice],
            non_blocking=self.k_centroids.is_pinned(),
        )
        destination_v[: segment.blocks].copy_(
            self.value_sums[block_slice],
            non_blocking=self.value_sums.is_pinned(),
        )
        stats.h2d_bytes += 2 * segment.blocks * destination_k.shape[1] * destination_k.shape[2] * 2
        return segment.blocks

    def load_kv(
        self,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        destination_k_scales: torch.Tensor,
        destination_v_scales: torch.Tensor,
        buffer_index: int,
        segment_id: int,
        local_start: int,
        local_stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> int:
        segment = self.segments[segment_id]
        token_start = segment.token_start + local_start
        token_stop = segment.token_start + local_stop
        blocks = math.ceil((local_stop - local_start) / SOL_BLOCK_TOKENS)
        block_start = segment.block_start + local_start // SOL_BLOCK_TOKENS
        block_stop = block_start + blocks
        assert self.k_scales is not None and self.v_scales is not None
        workspace = self.dense
        with torch.cuda.stream(workspace.h2d_stream):
            if workspace.kv_has_pending_compute[buffer_index]:
                workspace.h2d_stream.wait_event(workspace.kv_free[buffer_index])
            destination_k[: local_stop - local_start].copy_(
                self.k_quantized[token_start:token_stop],
                non_blocking=self.k_quantized.is_pinned(),
            )
            destination_v[: local_stop - local_start].copy_(
                self.v_quantized[token_start:token_stop],
                non_blocking=self.v_quantized.is_pinned(),
            )
            destination_k_scales[:blocks].copy_(
                self.k_scales[block_start:block_stop],
                non_blocking=self.k_scales.is_pinned(),
            )
            destination_v_scales[:blocks].copy_(
                self.v_scales[block_start:block_stop],
                non_blocking=self.v_scales.is_pinned(),
            )
            workspace.kv_ready[buffer_index].record(workspace.h2d_stream)
        compute_stream.wait_event(workspace.kv_ready[buffer_index])
        tokens = local_stop - local_start
        stats.h2d_bytes += 2 * (
            tokens * destination_k.shape[1] * destination_k.shape[2]
            + blocks * destination_k.shape[1] * 2
        )
        return blocks

    def release_q(self, compute_stream: torch.cuda.Stream) -> None:
        self.dense.q_free.record(compute_stream)
        self.dense.q_has_pending_compute = True

    def release_kv(self, buffer_index: int, compute_stream: torch.cuda.Stream) -> None:
        self.dense.kv_free[buffer_index].record(compute_stream)
        self.dense.kv_has_pending_compute[buffer_index] = True

    def recover(self) -> None:
        self.dense.recover()


class SolMaterializedQKVProducer:
    """Encode projected H3 Q/K/V into one reusable Sol materialized source."""

    def __init__(
        self,
        projected: ProjectedAttentionRunner,
        plan: SolStreamingPlan,
        dense: CudaWorkspace,
    ) -> None:
        self.projected = projected
        self.source = SolMaterializedSource(
            plan,
            dense,
            k_storage=projected.arena.k,
            v_storage=projected.arena.v,
            pin_memory=projected.pipeline_config.pin_qkv,
        )

    def materialize(
        self,
        hidden_host: torch.Tensor,
        cu_seqlens: torch.Tensor,
        project_qkv: QKVProjector,
        stats: ProjectedAttentionStats,
    ) -> SolMaterializedSource:
        bounds = validate_cu_seqlens(cu_seqlens, hidden_host.shape[0], "cu_seqlens")
        ranges = self.source.prepare(
            self.projected.arena.q,
            bounds,
            projection_tile_tokens=self.projected.pipeline_config.projection_tile_tokens,
        )
        self.projected.project_qkv_encoded_to_host(
            hidden_host,
            project_qkv,
            ranges=ranges,
            encode=self.source.encode,
            copy_to_host=self.source.copy_to_host,
            stats=stats,
        )
        stats.qkv_host_bytes = self.projected.arena.allocated_bytes + self.source.allocated_bytes
        return self.source


__all__ = ["SolMaterializedQKVProducer", "SolMaterializedSource"]
