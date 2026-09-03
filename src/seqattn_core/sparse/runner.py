from __future__ import annotations

import math
import time
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass
from itertools import pairwise

import torch

from .._single_flight import single_flight
from ..kernels import finalize_attention
from ..kernels.sol_preprocess import (
    SOL_BLOCK_TOKENS,
    compute_sol_thresholds,
)
from ..stats import StreamingAttentionStats
from ..streaming import StreamingAttentionRunner
from ..streaming.protocols import DeviceOutputConsumer
from ..streaming.tile_source import QKVTileSource
from ..streaming.workspace import CudaWorkspace
from ..validation import validate_cu_seqlens
from .materialized import SolMaterializedSource
from .plan import SolStreamingPlan, _validate_exact_prefix_tokens
from .transport import resolve_sol_transport


@dataclass
class SolStreamingStats(StreamingAttentionStats):
    kv_storage_dtype: str = ""
    summary_kv_tiles: int = 0
    summary_kv_tokens: int = 0
    precomputed_summary_blocks: int = 0
    summary_seconds: float = 0.0
    threshold_seconds: float = 0.0
    sparse_update_seconds: float = 0.0
    exact_route_blocks: int = 0
    approximate_route_blocks: int = 0

    @property
    def effective_density(self) -> float:
        total = self.exact_route_blocks + self.approximate_route_blocks
        return self.exact_route_blocks / total if total else 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        result = asdict(self)
        result["effective_density"] = self.effective_density
        return result


def _balanced_q_ranges(tokens: int, max_chunk_tokens: int) -> tuple[tuple[int, int], ...]:
    """Split a segment into route-aligned chunks without a small final Q tail."""

    chunks = math.ceil(tokens / max_chunk_tokens)
    route_blocks = math.ceil(tokens / SOL_BLOCK_TOKENS)
    blocks_per_chunk, larger_chunks = divmod(route_blocks, chunks)
    block_counts = [blocks_per_chunk] * (chunks - larger_chunks) + [
        blocks_per_chunk + 1
    ] * larger_chunks
    ranges = []
    start = 0
    for chunk_index, block_count in enumerate(block_counts):
        stop = tokens if chunk_index == chunks - 1 else start + block_count * SOL_BLOCK_TOKENS
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


class _SolStreamingWorkspace:
    def __init__(self, plan: SolStreamingPlan, dense: CudaWorkspace) -> None:
        attention = plan.attention
        self.plan = plan
        self.route_block_tokens = plan.route_block_tokens
        self.dense = dense
        device = attention.device
        summary_shape = (plan.max_kv_blocks, attention.q_heads, attention.head_dim)
        self.k_centroids = torch.empty(summary_shape, dtype=attention.dtype, device=device)
        self.value_sums = torch.empty_like(self.k_centroids)
        self.k_mean = torch.empty(
            (attention.q_heads, attention.head_dim), dtype=torch.float32, device=device
        )
        self.k_variance = torch.empty_like(self.k_mean)
        self.thresholds = torch.empty(
            (plan.max_q_blocks, attention.q_heads), dtype=torch.float32, device=device
        )
        self.route_counts = torch.empty(
            (plan.max_q_blocks, attention.q_heads, 2),
            dtype=torch.int32,
            device=device,
        )
        self.route_chunk_totals = torch.empty((2,), dtype=torch.int64, device=device)
        self.route_totals = torch.empty((2,), dtype=torch.int64, device=device)
        quantized_shape = (
            attention.kv_chunk_tokens,
            attention.kv_heads,
            attention.head_dim,
        )
        self.quantized_k = [
            torch.empty(quantized_shape, dtype=torch.int8, device=device)
            for _ in range(attention.num_kv_buffers)
        ]
        self.quantized_v = [torch.empty_like(tensor) for tensor in self.quantized_k]
        scale_shape = (
            math.ceil(attention.kv_chunk_tokens / SOL_BLOCK_TOKENS),
            attention.kv_heads,
        )
        self.k_scales = [
            torch.empty(scale_shape, dtype=torch.float16, device=device)
            for _ in range(attention.num_kv_buffers)
        ]
        self.v_scales = [torch.empty_like(tensor) for tensor in self.k_scales]

    def recover(self) -> None:
        self.dense.recover()


class SolStreamingAttentionRunner:
    """Bounded-HBM Sol attention sharing one exact streaming workspace."""

    def __init__(
        self,
        plan: SolStreamingPlan,
        dense_runner: StreamingAttentionRunner,
    ) -> None:
        if dense_runner.plan != plan.attention:
            raise ValueError("dense runner and sol_streaming plan must match")
        if dense_runner.backend != "triton":
            raise RuntimeError("Triton is not available for sol_streaming")
        capability = torch.cuda.get_device_capability(plan.attention.device)
        if capability[0] < 8:
            raise RuntimeError(
                "sol_streaming requires NVIDIA compute capability >= 8.0; "
                f"got SM{capability[0]}{capability[1]}"
            )
        runtime = dense_runner._borrow_cuda_runtime()
        self.plan = plan
        self.dense_runner = dense_runner
        self.workspace = _SolStreamingWorkspace(plan, runtime.workspace)
        self._single_flight_lock = runtime.single_flight_lock

    def _range(self, name: str):
        if self.plan.attention.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _prepare_stats(self, stats: SolStreamingStats | None) -> SolStreamingStats:
        stats = SolStreamingStats() if stats is None else stats
        stats.backend = "sol_streaming:triton"
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        stats.q_chunk_tokens = self.plan.attention.q_chunk_tokens
        stats.kv_chunk_tokens = self.plan.attention.kv_chunk_tokens
        return stats

    @single_flight
    @torch.inference_mode()
    def run_with_device_consumer(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        exact_prefix_tokens: tuple[int, ...],
        output_consumer: DeviceOutputConsumer,
        tau: float = 1.0,
        softmax_scale: float | None = None,
        stats: SolStreamingStats | None = None,
    ) -> None:
        source, q_bounds, k_bounds = self.dense_runner._prepare_host_qkv_source(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens,
            cu_seqlens,
        )
        if q_bounds != k_bounds:
            raise ValueError("sol_streaming requires self-attention with matching Q/KV segments")
        self._run_from_source(
            source,
            q_cpu.shape[0],
            cu_seqlens,
            exact_prefix_tokens=exact_prefix_tokens,
            output_consumer=output_consumer,
            tau=tau,
            softmax_scale=softmax_scale,
            stats=stats,
        )

    @single_flight
    @torch.inference_mode()
    def run_with_qkv_source(
        self,
        source: QKVTileSource | SolMaterializedSource,
        tokens: int,
        cu_seqlens: torch.Tensor,
        *,
        exact_prefix_tokens: tuple[int, ...],
        output_consumer: DeviceOutputConsumer,
        tau: float = 1.0,
        softmax_scale: float | None = None,
        stats: SolStreamingStats | None = None,
    ) -> None:
        self._run_from_source(
            source,
            tokens,
            cu_seqlens,
            exact_prefix_tokens=exact_prefix_tokens,
            output_consumer=output_consumer,
            tau=tau,
            softmax_scale=softmax_scale,
            stats=stats,
        )

    def _run_from_source(
        self,
        source: QKVTileSource | SolMaterializedSource,
        tokens: int,
        cu_seqlens: torch.Tensor,
        *,
        exact_prefix_tokens: tuple[int, ...],
        output_consumer: DeviceOutputConsumer,
        tau: float,
        softmax_scale: float | None,
        stats: SolStreamingStats | None,
    ) -> None:
        if tokens <= 0 or tokens > min(
            self.plan.attention.max_q_tokens,
            self.plan.attention.max_kv_tokens,
        ):
            raise ValueError("tokens must be positive and fit the sol_streaming plan")
        if not math.isfinite(tau):
            raise ValueError("tau must be finite")
        bounds = validate_cu_seqlens(cu_seqlens, tokens, "cu_seqlens")
        _validate_exact_prefix_tokens(exact_prefix_tokens, bounds)
        if isinstance(source, SolMaterializedSource):
            source.validate_layout(self.plan, bounds)
        stats = self._prepare_stats(stats)
        started = time.perf_counter()
        try:
            self._execute(
                source,
                bounds,
                exact_prefix_tokens,
                output_consumer,
                float(tau),
                self.plan.attention.head_dim**-0.5
                if softmax_scale is None
                else float(softmax_scale),
                stats,
            )
        except Exception:
            with suppress(Exception):
                output_consumer.synchronize()
            with suppress(Exception):
                source.recover()
            self.workspace.recover()
            raise
        stats.wall_seconds += time.perf_counter() - started

    def _execute(
        self,
        source: QKVTileSource | SolMaterializedSource,
        bounds: list[int],
        exact_prefix_tokens: tuple[int, ...],
        output_consumer: DeviceOutputConsumer,
        tau: float,
        scale: float,
        stats: SolStreamingStats,
    ) -> None:
        plan = self.plan.attention
        workspace = self.workspace
        dense = workspace.dense
        compute_stream = dense.compute_stream
        summary_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        threshold_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        update_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        transport = resolve_sol_transport(source, workspace, stats)
        stats.kv_storage_dtype = transport.storage_dtype
        stats.backend = f"sol_streaming:triton:{transport.storage_dtype}"

        with torch.cuda.device(plan.device):
            dense.pipeline_start.record(compute_stream)
            with torch.cuda.stream(compute_stream):
                workspace.route_totals.zero_()
                for segment_id, (segment_start, segment_stop) in enumerate(pairwise(bounds)):
                    segment_tokens = segment_stop - segment_start
                    if segment_tokens == 0:
                        continue
                    segment_blocks = math.ceil(segment_tokens / SOL_BLOCK_TOKENS)

                    summary_start = torch.cuda.Event(enable_timing=True)
                    summary_end = torch.cuda.Event(enable_timing=True)
                    summary_events.append((summary_start, summary_end))
                    summary_start.record(compute_stream)
                    with self._range("seqattn:sol_kv_summary"):
                        transport.prepare_segment(
                            segment_id=segment_id,
                            segment_start=segment_start,
                            segment_tokens=segment_tokens,
                            segment_blocks=segment_blocks,
                            compute_stream=compute_stream,
                        )
                    summary_end.record(compute_stream)

                    for q_local_start, q_local_stop in _balanced_q_ranges(
                        segment_tokens,
                        plan.q_chunk_tokens,
                    ):
                        q_tokens = q_local_stop - q_local_start
                        stats.q_chunks += 1
                        stats.max_resident_q_tokens = max(stats.max_resident_q_tokens, q_tokens)
                        source.load_q(
                            dense.q[:q_tokens],
                            segment_start + q_local_start,
                            segment_start + q_local_stop,
                            compute_stream,
                            stats,
                        )
                        threshold_start = torch.cuda.Event(enable_timing=True)
                        threshold_end = torch.cuda.Event(enable_timing=True)
                        threshold_events.append((threshold_start, threshold_end))
                        threshold_start.record(compute_stream)
                        with self._range("seqattn:sol_threshold"):
                            compute_sol_thresholds(
                                dense.q,
                                workspace.k_mean,
                                workspace.k_variance,
                                workspace.thresholds,
                                q_tokens=q_tokens,
                                softmax_scale=scale,
                                tau=tau,
                            )
                        threshold_end.record(compute_stream)

                        initialize = True
                        update_start = torch.cuda.Event(enable_timing=True)
                        update_end = torch.cuda.Event(enable_timing=True)
                        update_events.append((update_start, update_end))
                        update_start.record(compute_stream)
                        for tile_index, kv_local_start in enumerate(
                            range(0, segment_tokens, plan.kv_chunk_tokens)
                        ):
                            kv_local_stop = min(
                                kv_local_start + plan.kv_chunk_tokens,
                                segment_tokens,
                            )
                            with self._range("seqattn:sol_streaming_update"):
                                transport.update_tile(
                                    segment_id=segment_id,
                                    segment_start=segment_start,
                                    segment_tokens=segment_tokens,
                                    exact_prefix_tokens=exact_prefix_tokens[segment_id],
                                    q_tokens=q_tokens,
                                    q_block_offset=q_local_start // SOL_BLOCK_TOKENS,
                                    kv_local_start=kv_local_start,
                                    kv_local_stop=kv_local_stop,
                                    tile_index=tile_index,
                                    softmax_scale=scale,
                                    initialize=initialize,
                                    compute_stream=compute_stream,
                                )
                            initialize = False
                        update_end.record(compute_stream)
                        q_blocks = math.ceil(q_tokens / SOL_BLOCK_TOKENS)
                        torch.sum(
                            workspace.route_counts[:q_blocks],
                            dim=(0, 1),
                            dtype=torch.int64,
                            out=workspace.route_chunk_totals,
                        )
                        workspace.route_totals.add_(workspace.route_chunk_totals)

                        with self._range("seqattn:sol_finalize"):
                            finalize_attention(
                                dense.accumulator,
                                dense.running_sum,
                                dense.q,
                                q_tokens=q_tokens,
                            )
                        with self._range("seqattn:device_output_consumer"):
                            output_consumer(
                                dense.q[:q_tokens].reshape(q_tokens, -1),
                                segment_start + q_local_start,
                                segment_start + q_local_stop,
                            )
                        source.release_q(compute_stream)
                output_consumer.finish()
            dense.pipeline_end.record(compute_stream)
            output_consumer.synchronize()
            dense.pipeline_end.synchronize()
            counts = workspace.route_totals.cpu().tolist()
            stats.exact_route_blocks += int(counts[0])
            stats.approximate_route_blocks += int(counts[1])
            stats.compute_pipeline_seconds += (
                dense.pipeline_start.elapsed_time(dense.pipeline_end) / 1000.0
            )
            stats.summary_seconds += (
                sum(start.elapsed_time(end) for start, end in summary_events) / 1000.0
            )
            stats.threshold_seconds += (
                sum(start.elapsed_time(end) for start, end in threshold_events) / 1000.0
            )
            stats.sparse_update_seconds += (
                sum(start.elapsed_time(end) for start, end in update_events) / 1000.0
            )


__all__ = ["SolStreamingAttentionRunner", "SolStreamingStats"]
