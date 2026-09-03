from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from ..kernels.sol_preprocess import compute_sol_k_stats, summarize_sol_kv
from ..kernels.sol_streaming import (
    update_sol_attention_state,
    update_sol_attention_state_int8,
)
from ..streaming.tile_source import QKVTileSource
from .materialized import SolMaterializedSource

if TYPE_CHECKING:
    from .runner import SolStreamingStats, _SolStreamingWorkspace


class SolTransport(Protocol):
    storage_dtype: str

    def prepare_segment(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        segment_blocks: int,
        compute_stream: torch.cuda.Stream,
    ) -> None: ...

    def update_tile(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        exact_prefix_tokens: int,
        q_tokens: int,
        q_block_offset: int,
        kv_local_start: int,
        kv_local_stop: int,
        tile_index: int,
        softmax_scale: float,
        initialize: bool,
        compute_stream: torch.cuda.Stream,
    ) -> None: ...


class _Bf16Transport:
    storage_dtype = "bf16"

    def __init__(
        self,
        source: QKVTileSource,
        workspace: _SolStreamingWorkspace,
        stats: SolStreamingStats,
    ) -> None:
        self.source = source
        self.workspace = workspace
        self.stats = stats
        self.plan = workspace.plan.attention

    def prepare_segment(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        segment_blocks: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        del segment_id
        dense = self.workspace.dense
        summary_block_offset = 0
        for tile_index, local_start in enumerate(
            range(0, segment_tokens, self.plan.kv_chunk_tokens)
        ):
            local_stop = min(local_start + self.plan.kv_chunk_tokens, segment_tokens)
            kv_tokens = local_stop - local_start
            buffer_index = tile_index % self.plan.num_kv_buffers
            self.source.load_kv(
                dense.k[buffer_index][:kv_tokens],
                dense.v[buffer_index][:kv_tokens],
                buffer_index,
                segment_start + local_start,
                segment_start + local_stop,
                compute_stream,
                self.stats,
            )
            summary_block_offset += summarize_sol_kv(
                dense.k[buffer_index],
                dense.v[buffer_index],
                self.workspace.k_centroids,
                self.workspace.value_sums,
                kv_tokens=kv_tokens,
                summary_block_offset=summary_block_offset,
            )
            self.source.release_kv(buffer_index, compute_stream)
            self.stats.summary_kv_tiles += 1
            self.stats.summary_kv_tokens += kv_tokens
        if summary_block_offset != segment_blocks:
            raise RuntimeError("sol_streaming summary block accounting failed")
        compute_sol_k_stats(
            self.workspace.k_centroids,
            self.workspace.k_mean,
            self.workspace.k_variance,
            num_blocks=segment_blocks,
        )

    def update_tile(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        exact_prefix_tokens: int,
        q_tokens: int,
        q_block_offset: int,
        kv_local_start: int,
        kv_local_stop: int,
        tile_index: int,
        softmax_scale: float,
        initialize: bool,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        del segment_id
        dense = self.workspace.dense
        kv_tokens = kv_local_stop - kv_local_start
        buffer_index = tile_index % self.plan.num_kv_buffers
        self.source.load_kv(
            dense.k[buffer_index][:kv_tokens],
            dense.v[buffer_index][:kv_tokens],
            buffer_index,
            segment_start + kv_local_start,
            segment_start + kv_local_stop,
            compute_stream,
            self.stats,
        )
        update_sol_attention_state(
            dense.q,
            dense.k[buffer_index],
            dense.v[buffer_index],
            self.workspace.k_centroids,
            self.workspace.value_sums,
            self.workspace.thresholds,
            dense.running_max,
            dense.running_sum,
            dense.accumulator,
            self.workspace.route_counts,
            q_tokens=q_tokens,
            kv_tokens=kv_tokens,
            q_block_offset=q_block_offset,
            kv_block_offset=kv_local_start // self.workspace.route_block_tokens,
            segment_tokens=segment_tokens,
            exact_prefix_tokens=exact_prefix_tokens,
            softmax_scale=softmax_scale,
            initialize=initialize,
        )
        self.source.release_kv(buffer_index, compute_stream)
        self.stats.kv_tiles += 1


class _Int8Transport:
    storage_dtype = "int8"

    def __init__(
        self,
        source: SolMaterializedSource,
        workspace: _SolStreamingWorkspace,
        stats: SolStreamingStats,
    ) -> None:
        self.source = source
        self.workspace = workspace
        self.stats = stats
        self.plan = workspace.plan.attention

    def prepare_segment(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        segment_blocks: int,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        del segment_start, compute_stream
        loaded = self.source.load_summary(
            self.workspace.k_centroids,
            self.workspace.value_sums,
            segment_id,
            self.stats,
        )
        if loaded != segment_blocks:
            raise RuntimeError("precomputed Sol summary block accounting failed")
        compute_sol_k_stats(
            self.workspace.k_centroids,
            self.workspace.k_mean,
            self.workspace.k_variance,
            num_blocks=segment_blocks,
        )
        self.stats.summary_kv_tokens += segment_tokens
        self.stats.precomputed_summary_blocks += segment_blocks

    def update_tile(
        self,
        *,
        segment_id: int,
        segment_start: int,
        segment_tokens: int,
        exact_prefix_tokens: int,
        q_tokens: int,
        q_block_offset: int,
        kv_local_start: int,
        kv_local_stop: int,
        tile_index: int,
        softmax_scale: float,
        initialize: bool,
        compute_stream: torch.cuda.Stream,
    ) -> None:
        del segment_start
        dense = self.workspace.dense
        kv_tokens = kv_local_stop - kv_local_start
        buffer_index = tile_index % self.plan.num_kv_buffers
        self.source.load_kv(
            self.workspace.quantized_k[buffer_index],
            self.workspace.quantized_v[buffer_index],
            self.workspace.k_scales[buffer_index],
            self.workspace.v_scales[buffer_index],
            buffer_index,
            segment_id,
            kv_local_start,
            kv_local_stop,
            compute_stream,
            self.stats,
        )
        update_sol_attention_state_int8(
            dense.q,
            self.workspace.quantized_k[buffer_index],
            self.workspace.quantized_v[buffer_index],
            self.workspace.k_scales[buffer_index],
            self.workspace.v_scales[buffer_index],
            self.workspace.k_centroids,
            self.workspace.value_sums,
            self.workspace.thresholds,
            dense.running_max,
            dense.running_sum,
            dense.accumulator,
            self.workspace.route_counts,
            q_tokens=q_tokens,
            kv_tokens=kv_tokens,
            q_block_offset=q_block_offset,
            kv_block_offset=kv_local_start // self.workspace.route_block_tokens,
            segment_tokens=segment_tokens,
            exact_prefix_tokens=exact_prefix_tokens,
            softmax_scale=softmax_scale,
            initialize=initialize,
        )
        self.source.release_kv(buffer_index, compute_stream)
        self.stats.kv_tiles += 1


def resolve_sol_transport(
    source: QKVTileSource | SolMaterializedSource,
    workspace: _SolStreamingWorkspace,
    stats: SolStreamingStats,
) -> SolTransport:
    if isinstance(source, SolMaterializedSource):
        return _Int8Transport(source, workspace, stats)
    return _Bf16Transport(source, workspace, stats)


__all__ = ["SolTransport", "resolve_sol_transport"]
