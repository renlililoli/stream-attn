from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .cache import CacheLookup, KVPageCache
from .config import PagedAttentionConfig
from .host_memory import HostMemoryPlan, HostMemorySnapshot
from .kernels import (
    finalize_attention,
    update_attention_state,
    update_attention_state_int8,
)
from .paging import (
    KVLayout,
    PageDescriptor,
    PageReadMetrics,
    PageReader,
    PageSink,
    PageSource,
    PageWriter,
    TensorLayout,
    pages_by_segment,
    validate_cu_seqlens,
)
from .planner import AttentionPlan, build_plan
from .quantization import dequantize_int8_per_token_group
from .runtime import resolve_backend
from .stats import PagedAttentionStats


@dataclass
class _KVStage:
    k: torch.Tensor
    v: torch.Tensor
    k_scales: torch.Tensor | None
    v_scales: torch.Tensor | None


@dataclass(frozen=True)
class _LoadedPage:
    page: PageDescriptor
    stage_index: int
    metrics: PageReadMetrics
    cache: CacheLookup


class _HostStaging:
    def __init__(
        self,
        q_layout: TensorLayout,
        kv_layout: KVLayout,
        q_pages: Sequence[PageDescriptor],
        kv_pages: Sequence[PageDescriptor],
        *,
        queue_depth: int,
        output_buffers: int,
        pinned: bool,
        memory_plan: HostMemoryPlan,
    ) -> None:
        self.memory_plan = memory_plan
        self.registered_bytes = 0
        max_q_tokens = max((page.padded_tokens for page in q_pages), default=1)
        max_kv_tokens = max((page.padded_tokens for page in kv_pages), default=1)
        self.q = torch.empty(
            (max_q_tokens, q_layout.heads, q_layout.head_dim),
            dtype=q_layout.torch_dtype,
            pin_memory=pinned,
        )
        self.outputs = [
            torch.empty_like(self.q, pin_memory=pinned) for _ in range(output_buffers)
        ]
        self.kv: list[_KVStage] = []
        scale_groups = math.ceil(max_kv_tokens / kv_layout.quant_group_tokens)
        for _ in range(queue_depth):
            k = torch.empty(
                (max_kv_tokens, kv_layout.heads, kv_layout.head_dim),
                dtype=kv_layout.storage_torch_dtype,
                pin_memory=pinned,
            )
            v = torch.empty_like(k, pin_memory=pinned)
            if kv_layout.storage_dtype == "int8":
                k_scales = torch.empty(
                    (scale_groups, kv_layout.heads),
                    dtype=torch.float16,
                    pin_memory=pinned,
                )
                v_scales = torch.empty_like(k_scales, pin_memory=pinned)
            else:
                k_scales = v_scales = None
            self.kv.append(_KVStage(k, v, k_scales, v_scales))
        tensors = [self.q, *self.outputs]
        for stage in self.kv:
            tensors.extend((stage.k, stage.v))
            if stage.k_scales is not None and stage.v_scales is not None:
                tensors.extend((stage.k_scales, stage.v_scales))
        self.registered_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        try:
            memory_plan.register("pinned", self.registered_bytes)
        except BaseException:
            self.q = torch.empty(0)
            self.outputs = []
            self.kv = []
            raise

    def close(self) -> None:
        if not self.registered_bytes:
            return
        self.q = torch.empty(0)
        self.outputs = []
        self.kv = []
        self.memory_plan.release("pinned", self.registered_bytes)
        self.registered_bytes = 0


class _PagedCudaWorkspace:
    def __init__(self, plan: AttentionPlan, kv_layout: KVLayout) -> None:
        device = plan.device
        self.q = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device=device,
        )
        self.k = [
            torch.empty(
                (plan.kv_chunk_tokens, plan.kv_heads, plan.head_dim),
                dtype=kv_layout.storage_torch_dtype,
                device=device,
            )
            for _ in range(plan.num_kv_buffers)
        ]
        self.v = [torch.empty_like(tensor) for tensor in self.k]
        if kv_layout.storage_dtype == "int8":
            groups = math.ceil(
                (plan.kv_chunk_tokens + kv_layout.quant_group_tokens - 1)
                / kv_layout.quant_group_tokens
            )
            self.k_scales = [
                torch.empty((groups, plan.kv_heads), dtype=torch.float16, device=device)
                for _ in self.k
            ]
            self.v_scales = [torch.empty_like(tensor) for tensor in self.k_scales]
        else:
            self.k_scales = self.v_scales = None
        self.running_max = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads), dtype=torch.float32, device=device
        )
        self.running_sum = torch.empty_like(self.running_max)
        self.accumulator = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.output = [
            torch.empty_like(self.q) for _ in range(plan.num_output_buffers)
        ]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.q_ready = torch.cuda.Event()
        self.q_free = torch.cuda.Event()
        self.q_busy = False
        self.kv_ready = [torch.cuda.Event() for _ in self.k]
        self.kv_free = [torch.cuda.Event() for _ in self.k]
        self.kv_busy = [False for _ in self.k]
        self.output_ready = [torch.cuda.Event() for _ in self.output]
        self.output_free = [torch.cuda.Event() for _ in self.output]
        self.output_busy = [False for _ in self.output]
        self.stage_free: list[torch.cuda.Event] = []
        self.stage_busy: list[bool] = []
        self.output_host_ready: list[torch.cuda.Event] = []

    def bind_host_rings(self, staging: _HostStaging) -> None:
        self.stage_free = [torch.cuda.Event() for _ in staging.kv]
        self.stage_busy = [False for _ in staging.kv]
        self.output_host_ready = [torch.cuda.Event() for _ in staging.outputs]


def _backing_is_nvme(source: PageSource) -> bool:
    return getattr(source, "backing_kind", "memory") == "nvme"


def _validate_source_contract(
    q_source: PageSource,
    kv_source: PageSource,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    config: PagedAttentionConfig,
) -> tuple[TensorLayout, KVLayout, list[int], list[int]]:
    if q_source.q_layout is None:
        raise ValueError("q_source does not provide query pages")
    if kv_source.kv_layout is None:
        raise ValueError("kv_source does not provide K/V pages")
    q_layout = q_source.q_layout
    kv_layout = kv_source.kv_layout
    q_bounds = validate_cu_seqlens(cu_seqlens_q, q_layout.total_tokens, "cu_seqlens_q")
    k_bounds = validate_cu_seqlens(cu_seqlens_k, kv_layout.total_tokens, "cu_seqlens_k")
    if len(q_bounds) != len(k_bounds):
        raise ValueError("query and K/V sources must describe the same batch size")
    if q_source.cu_seqlens_q is not None and tuple(q_bounds) != q_source.cu_seqlens_q:
        raise ValueError("cu_seqlens_q does not match q_source")
    if kv_source.cu_seqlens_k is not None and tuple(k_bounds) != kv_source.cu_seqlens_k:
        raise ValueError("cu_seqlens_k does not match kv_source")
    if q_layout.head_dim != kv_layout.head_dim or q_layout.heads % kv_layout.heads:
        raise ValueError("query and K/V head layouts are incompatible")
    if q_layout.dtype != kv_layout.source_dtype:
        raise ValueError("query and K/V source dtype must match")
    if kv_layout.storage_dtype != config.kv_storage_dtype:
        raise ValueError(
            "kv_source storage dtype does not match PagedAttentionConfig: "
            f"{kv_layout.storage_dtype} != {config.kv_storage_dtype}"
        )
    for index, ((q_start, q_stop), (k_start, k_stop)) in enumerate(
        zip(zip(q_bounds[:-1], q_bounds[1:]), zip(k_bounds[:-1], k_bounds[1:]))
    ):
        if q_stop > q_start and k_stop == k_start:
            raise ValueError(f"sequence {index} has queries but no keys")
    for source in {id(q_source): q_source, id(kv_source): kv_source}.values():
        if _backing_is_nvme(source) and source.direct_io != config.direct_io:
            raise ValueError(
                "NVMe source direct_io mode must match PagedAttentionConfig; "
                "buffered fallback is never implicit"
            )
    return q_layout, kv_layout, q_bounds, k_bounds


def _load_kv_page(
    reader: PageReader,
    cache: KVPageCache,
    page: PageDescriptor,
    stage: _KVStage,
) -> tuple[PageReadMetrics, CacheLookup]:
    lookup = cache.get(page, stage.k, stage.v, stage.k_scales, stage.v_scales)
    if lookup.hit:
        return PageReadMetrics(), lookup
    metrics = reader.read_kv(
        page,
        stage.k,
        stage.v,
        stage.k_scales,
        stage.v_scales,
    )
    cache.put(page, stage.k, stage.v, stage.k_scales, stage.v_scales)
    return metrics, lookup


class PagedAttentionRunner:
    """Fixed-host-memory paged attention runtime for memory or NVMe sources."""

    def __init__(
        self,
        config: PagedAttentionConfig | None = None,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = PagedAttentionConfig() if config is None else config
        self.device = torch.device(device)

    @torch.no_grad()
    def run(
        self,
        q_source: PageSource,
        kv_source: PageSource,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        output_sink: PageSink,
        *,
        config: PagedAttentionConfig | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: PagedAttentionStats | None = None,
    ) -> object:
        config = self.config if config is None else config
        config.validate()
        if (
            getattr(output_sink, "backing_kind", "memory") == "nvme"
            and getattr(output_sink, "direct_io", None) != config.direct_io
        ):
            raise ValueError(
                "NVMe output sink direct_io mode must match PagedAttentionConfig"
            )
        q_layout, kv_layout, q_bounds, k_bounds = _validate_source_contract(
            q_source, kv_source, cu_seqlens_q, cu_seqlens_k, config
        )
        attention_config = config.attention
        if attention_config.output_mode != "host":
            raise ValueError("paged output sinks require attention output_mode='host'")
        plan = build_plan(
            q_heads=q_layout.heads,
            kv_heads=kv_layout.heads,
            head_dim=q_layout.head_dim,
            dtype=q_layout.torch_dtype,
            device=self.device,
            max_q_tokens=q_layout.total_tokens,
            max_kv_tokens=kv_layout.total_tokens,
            config=attention_config,
        )
        backend = resolve_backend(attention_config.backend, plan.dtype, plan.device)
        if backend == "reference" and plan.device.type != "cpu":
            raise ValueError("the paged reference backend requires device='cpu'")
        if backend == "triton" and kv_layout.storage_dtype == "fp32":
            raise ValueError("the Triton paged backend does not support FP32 K/V storage")

        stats = PagedAttentionStats() if stats is None else stats
        stats.backend = backend
        stats.kv_storage_dtype = kv_layout.storage_dtype
        stats.q_chunk_tokens = plan.q_chunk_tokens
        stats.kv_chunk_tokens = plan.kv_chunk_tokens
        stats.estimated_workspace_bytes = plan.estimated_workspace_bytes
        scale = plan.head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        q_groups = pages_by_segment(q_source.q_pages, len(q_bounds) - 1)
        kv_groups = pages_by_segment(kv_source.kv_pages, len(k_bounds) - 1)

        memory_plan = HostMemoryPlan(
            total_budget_bytes=config.host_memory_budget_bytes,
            pinned_limit_bytes=config.pinned_staging_budget_bytes,
            bounce_limit_bytes=config.direct_io_bounce_budget_bytes,
            metadata_margin_bytes=config.metadata_margin_bytes,
        )
        pinned = backend == "triton"
        staging = _HostStaging(
            q_layout,
            kv_layout,
            q_source.q_pages,
            kv_source.kv_pages,
            queue_depth=config.io_queue_depth,
            output_buffers=config.num_output_buffers,
            pinned=pinned,
            memory_plan=memory_plan,
        )
        q_reader: PageReader | None = None
        kv_reader: PageReader | None = None
        writer: PageWriter | None = None
        cache: KVPageCache | None = None
        executor: ThreadPoolExecutor | None = None
        output_futures: list[Future[PageReadMetrics] | None] = [
            None for _ in staging.outputs
        ]
        output_is_nvme = getattr(output_sink, "backing_kind", "memory") == "nvme"
        completed_snapshot: HostMemorySnapshot | None = None
        started = time.perf_counter()
        try:
            if q_source is kv_source:
                q_reader = kv_reader = q_source.open_reader(
                    memory_plan, config.io_queue_depth
                )
            else:
                q_reader = q_source.open_reader(memory_plan, 1)
                kv_reader = kv_source.open_reader(memory_plan, config.io_queue_depth)
            writer = output_sink.open_writer(
                q_layout,
                q_bounds,
                q_source.q_pages,
                memory_plan,
                config.num_output_buffers,
            )
            cache_capacity = min(
                memory_plan.cache_limit_bytes,
                max(0, memory_plan.total_budget_bytes - memory_plan.current_bytes),
            )
            cache = KVPageCache(
                kv_source.kv_pages,
                kv_layout,
                capacity_bytes=cache_capacity,
                hot_fraction=config.cache_hot_fraction,
                memory_plan=memory_plan,
            )
            executor = ThreadPoolExecutor(
                max_workers=config.io_workers,
                thread_name_prefix="seqattn-io",
            )
            if backend == "reference":
                self._run_reference(
                    plan,
                    q_reader,
                    kv_reader,
                    writer,
                    executor,
                    cache,
                    staging,
                    q_groups,
                    kv_groups,
                    q_bounds,
                    k_bounds,
                    scale,
                    causal,
                    stats,
                    output_futures,
                    _backing_is_nvme(q_source),
                    _backing_is_nvme(kv_source),
                    output_is_nvme,
                )
            else:
                self._run_triton(
                    plan,
                    kv_layout,
                    q_reader,
                    kv_reader,
                    writer,
                    executor,
                    cache,
                    staging,
                    q_groups,
                    kv_groups,
                    q_bounds,
                    k_bounds,
                    scale,
                    causal,
                    stats,
                    output_futures,
                    _backing_is_nvme(q_source),
                    _backing_is_nvme(kv_source),
                    output_is_nvme,
                )
            self._finish_output_futures(output_futures, stats, output_is_nvme)
            result = writer.close()
            writer = None
            stats.cache_hits = cache.hits
            stats.cache_misses = cache.misses
            lookups = cache.hits + cache.misses
            stats.cache_hit_ratio = cache.hits / lookups if lookups else 0.0
            completed_snapshot = memory_plan.snapshot()
            stats.wall_seconds += time.perf_counter() - started
            self._record_memory_stats(stats, completed_snapshot)
            return result
        except BaseException:
            for future in output_futures:
                if future is not None:
                    future.cancel()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            if writer is not None:
                writer.abort()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if cache is not None:
                cache.close()
            if kv_reader is not None:
                kv_reader.close()
            if q_reader is not None and q_reader is not kv_reader:
                q_reader.close()
            staging.close()
            snapshot = memory_plan.snapshot()
            self._record_memory_stats(stats, completed_snapshot or snapshot)

    @staticmethod
    def _record_memory_stats(
        stats: PagedAttentionStats, snapshot: HostMemorySnapshot
    ) -> None:
        stats.operator_host_allocated_bytes = snapshot.operator_host_allocated_bytes
        stats.operator_host_peak_bytes = snapshot.operator_host_peak_bytes
        stats.pinned_peak_bytes = snapshot.pinned_peak_bytes
        stats.direct_io_bounce_peak_bytes = snapshot.direct_io_bounce_peak_bytes
        stats.dram_cache_peak_bytes = snapshot.dram_cache_peak_bytes
        stats.host_memory_budget_bytes = snapshot.host_memory_budget_bytes

    @staticmethod
    def _accumulate_read(
        stats: PagedAttentionStats,
        metrics: PageReadMetrics,
        lookup: CacheLookup | None,
        is_nvme: bool,
    ) -> None:
        if lookup is not None:
            stats.cache_lookup_seconds += lookup.seconds
        stats.quantization_seconds += metrics.quantization_seconds
        if is_nvme:
            stats.nvme_read_seconds += metrics.read_seconds
            stats.nvme_logical_read_bytes += metrics.logical_bytes
            stats.nvme_physical_read_bytes += metrics.physical_bytes

    @staticmethod
    def _finish_output_futures(
        futures: Sequence[Future[PageReadMetrics] | None],
        stats: PagedAttentionStats,
        output_is_nvme: bool,
    ) -> None:
        for future in futures:
            if future is None:
                continue
            started = time.perf_counter()
            metrics = future.result()
            stats.io_queue_wait_seconds += time.perf_counter() - started
            stats.output_write_seconds += metrics.read_seconds
            if output_is_nvme:
                stats.nvme_write_seconds += metrics.read_seconds
                stats.nvme_logical_write_bytes += metrics.logical_bytes
                stats.nvme_physical_write_bytes += metrics.physical_bytes

    def _submit_output(
        self,
        page: PageDescriptor,
        data: torch.Tensor,
        slot: int,
        writer: PageWriter,
        executor: ThreadPoolExecutor,
        futures: list[Future[PageReadMetrics] | None],
        stats: PagedAttentionStats,
        output_is_nvme: bool,
    ) -> None:
        previous = futures[slot]
        if previous is not None:
            started = time.perf_counter()
            metrics = previous.result()
            stats.io_queue_wait_seconds += time.perf_counter() - started
            stats.output_write_seconds += metrics.read_seconds
            if output_is_nvme:
                stats.nvme_write_seconds += metrics.read_seconds
                stats.nvme_logical_write_bytes += metrics.logical_bytes
                stats.nvme_physical_write_bytes += metrics.physical_bytes
        futures[slot] = executor.submit(writer.write_page, page, data[: page.valid_tokens])

    def _scan_kv_pages(
        self,
        pages: Sequence[PageDescriptor],
        stages: Sequence[_KVStage],
        reader: PageReader,
        executor: ThreadPoolExecutor,
        cache: KVPageCache,
        consume: Callable[[_LoadedPage, _KVStage], None],
        stats: PagedAttentionStats,
        *,
        source_is_nvme: bool,
        stage_reuse_wait: Callable[[int], None] | None = None,
    ) -> None:
        depth = min(len(stages), len(pages))
        pending: dict[int, Future[tuple[PageReadMetrics, CacheLookup]]] = {}

        def submit(index: int) -> None:
            slot = index % len(stages)
            if stage_reuse_wait is not None:
                stage_reuse_wait(slot)
            pending[index] = executor.submit(
                _load_kv_page,
                reader,
                cache,
                pages[index],
                stages[slot],
            )

        for index in range(depth):
            submit(index)
        for index, page in enumerate(pages):
            wait_started = time.perf_counter()
            metrics, lookup = pending.pop(index).result()
            stats.io_queue_wait_seconds += time.perf_counter() - wait_started
            self._accumulate_read(stats, metrics, lookup, source_is_nvme)
            slot = index % len(stages)
            loaded = _LoadedPage(page, slot, metrics, lookup)
            consume(loaded, stages[slot])
            stats.kv_pages += 1
            next_index = index + depth
            if next_index < len(pages):
                submit(next_index)

    def _run_reference(
        self,
        plan: AttentionPlan,
        q_reader: PageReader,
        kv_reader: PageReader,
        writer: PageWriter,
        executor: ThreadPoolExecutor,
        cache: KVPageCache,
        staging: _HostStaging,
        q_groups: Sequence[Sequence[PageDescriptor]],
        kv_groups: Sequence[Sequence[PageDescriptor]],
        q_bounds: Sequence[int],
        k_bounds: Sequence[int],
        scale: float,
        causal: bool,
        stats: PagedAttentionStats,
        output_futures: list[Future[PageReadMetrics] | None],
        q_is_nvme: bool,
        kv_is_nvme: bool,
        output_is_nvme: bool = False,
    ) -> None:
        group_size = plan.q_heads // plan.kv_heads
        for segment_id, (q_pages, kv_pages) in enumerate(zip(q_groups, kv_groups)):
            q_length = q_bounds[segment_id + 1] - q_bounds[segment_id]
            k_length = k_bounds[segment_id + 1] - k_bounds[segment_id]
            causal_shift = k_length - q_length
            for q_page in q_pages:
                output_slot = q_page.page_id % len(staging.outputs)
                output = staging.outputs[output_slot]
                metrics = q_reader.read_q(q_page, staging.q)
                self._accumulate_read(stats, metrics, None, q_is_nvme)
                stats.q_pages += 1
                for page_offset in range(0, q_page.valid_tokens, plan.q_chunk_tokens):
                    q_tokens = min(plan.q_chunk_tokens, q_page.valid_tokens - page_offset)
                    q_local_offset = q_page.segment_token_start + page_offset
                    q = staging.q[page_offset : page_offset + q_tokens].transpose(0, 1).float()
                    running_max = torch.full(
                        (plan.q_heads, q_tokens), -math.inf, dtype=torch.float32
                    )
                    running_sum = torch.zeros_like(running_max)
                    accumulator = torch.zeros(
                        (plan.q_heads, q_tokens, plan.head_dim), dtype=torch.float32
                    )
                    q_positions = torch.arange(q_local_offset, q_local_offset + q_tokens)
                    stats.q_chunks += 1
                    stats.kv_page_scans += 1

                    def consume(loaded: _LoadedPage, stage: _KVStage) -> None:
                        page = loaded.page
                        if plan.dtype == torch.int8:
                            raise AssertionError("query dtype cannot be int8")
                        if stage.k.dtype == torch.int8:
                            assert stage.k_scales is not None and stage.v_scales is not None
                            groups = math.ceil(page.valid_tokens / 64)
                            k_page = dequantize_int8_per_token_group(
                                stage.k[: page.valid_tokens],
                                stage.k_scales[:groups],
                                dtype=plan.dtype,
                            )
                            v_page = dequantize_int8_per_token_group(
                                stage.v[: page.valid_tokens],
                                stage.v_scales[:groups],
                                dtype=plan.dtype,
                            )
                        else:
                            k_page = stage.k[: page.valid_tokens]
                            v_page = stage.v[: page.valid_tokens]
                        for kv_offset in range(0, page.valid_tokens, plan.kv_chunk_tokens):
                            kv_tokens = min(plan.kv_chunk_tokens, page.valid_tokens - kv_offset)
                            k = k_page[kv_offset : kv_offset + kv_tokens]
                            v = v_page[kv_offset : kv_offset + kv_tokens]
                            if group_size != 1:
                                k = k.repeat_interleave(group_size, dim=1)
                                v = v.repeat_interleave(group_size, dim=1)
                            k = k.transpose(0, 1).float()
                            v = v.transpose(0, 1).float()
                            kernel_started = time.perf_counter()
                            scores = torch.matmul(q, k.transpose(-1, -2)).mul_(scale)
                            if causal:
                                kv_local = page.segment_token_start + kv_offset
                                k_positions = torch.arange(kv_local, kv_local + kv_tokens)
                                valid = k_positions.unsqueeze(0) <= (
                                    q_positions.unsqueeze(1) + causal_shift
                                )
                                scores.masked_fill_(~valid.unsqueeze(0), -math.inf)
                            tile_max = scores.amax(dim=-1)
                            merged_max = torch.maximum(running_max, tile_max)
                            valid_rows = torch.isfinite(merged_max)
                            old_scale = torch.where(
                                valid_rows,
                                torch.exp(running_max - merged_max),
                                torch.ones_like(merged_max),
                            )
                            probabilities = torch.where(
                                torch.isfinite(scores),
                                torch.exp(scores - merged_max.unsqueeze(-1)),
                                torch.zeros_like(scores),
                            )
                            running_sum.mul_(old_scale).add_(probabilities.sum(dim=-1))
                            accumulator.mul_(old_scale.unsqueeze(-1)).add_(
                                torch.matmul(probabilities, v)
                            )
                            running_max.copy_(torch.where(valid_rows, merged_max, running_max))
                            stats.gpu_kernel_seconds += time.perf_counter() - kernel_started
                            stats.state_spills += 1

                    self._scan_kv_pages(
                        kv_pages,
                        staging.kv,
                        kv_reader,
                        executor,
                        cache,
                        consume,
                        stats,
                        source_is_nvme=kv_is_nvme,
                    )
                    normalized = torch.where(
                        running_sum.unsqueeze(-1) > 0,
                        accumulator / running_sum.unsqueeze(-1),
                        torch.zeros_like(accumulator),
                    )
                    output[page_offset : page_offset + q_tokens].copy_(
                        normalized.transpose(0, 1).to(plan.dtype)
                    )
                self._submit_output(
                    q_page,
                    output,
                    output_slot,
                    writer,
                    executor,
                    output_futures,
                    stats,
                    output_is_nvme,
                )

    def _run_triton(
        self,
        plan: AttentionPlan,
        kv_layout: KVLayout,
        q_reader: PageReader,
        kv_reader: PageReader,
        writer: PageWriter,
        executor: ThreadPoolExecutor,
        cache: KVPageCache,
        staging: _HostStaging,
        q_groups: Sequence[Sequence[PageDescriptor]],
        kv_groups: Sequence[Sequence[PageDescriptor]],
        q_bounds: Sequence[int],
        k_bounds: Sequence[int],
        scale: float,
        causal: bool,
        stats: PagedAttentionStats,
        output_futures: list[Future[PageReadMetrics] | None],
        q_is_nvme: bool,
        kv_is_nvme: bool,
        output_is_nvme: bool = False,
    ) -> None:
        workspace = _PagedCudaWorkspace(plan, kv_layout)
        workspace.bind_host_rings(staging)
        q_chunk_index = 0
        kernel_start = torch.cuda.Event(enable_timing=True)
        kernel_end = torch.cuda.Event(enable_timing=True)
        kernel_started = False
        with torch.cuda.device(plan.device):
            for segment_id, (q_pages, kv_pages) in enumerate(zip(q_groups, kv_groups)):
                q_length = q_bounds[segment_id + 1] - q_bounds[segment_id]
                k_length = k_bounds[segment_id + 1] - k_bounds[segment_id]
                causal_shift = k_length - q_length
                for q_page in q_pages:
                    output_host_slot = q_page.page_id % len(staging.outputs)
                    previous = output_futures[output_host_slot]
                    if previous is not None:
                        wait_started = time.perf_counter()
                        metrics = previous.result()
                        stats.io_queue_wait_seconds += time.perf_counter() - wait_started
                        stats.output_write_seconds += metrics.read_seconds
                        if output_is_nvme:
                            stats.nvme_write_seconds += metrics.read_seconds
                            stats.nvme_logical_write_bytes += metrics.logical_bytes
                            stats.nvme_physical_write_bytes += metrics.physical_bytes
                        output_futures[output_host_slot] = None
                    metrics = q_reader.read_q(q_page, staging.q)
                    self._accumulate_read(stats, metrics, None, q_is_nvme)
                    stats.q_pages += 1
                    for page_offset in range(0, q_page.valid_tokens, plan.q_chunk_tokens):
                        q_tokens = min(plan.q_chunk_tokens, q_page.valid_tokens - page_offset)
                        q_local_offset = q_page.segment_token_start + page_offset
                        output_index = q_chunk_index % len(workspace.output)
                        with torch.cuda.stream(workspace.h2d_stream):
                            if workspace.q_busy:
                                workspace.h2d_stream.wait_event(workspace.q_free)
                            copy_started = time.perf_counter()
                            workspace.q[:q_tokens].copy_(
                                staging.q[page_offset : page_offset + q_tokens],
                                non_blocking=True,
                            )
                            stats.pinned_copy_seconds += time.perf_counter() - copy_started
                            workspace.q_ready.record(workspace.h2d_stream)
                        stats.h2d_bytes += q_tokens * q_layout_bytes(plan)
                        stats.q_chunks += 1
                        stats.kv_page_scans += 1
                        with torch.cuda.stream(workspace.compute_stream):
                            workspace.compute_stream.wait_event(workspace.q_ready)
                            if not kernel_started:
                                kernel_start.record(workspace.compute_stream)
                                kernel_started = True
                        initialize = True
                        kv_tile_index = 0

                        def wait_stage(slot: int) -> None:
                            if workspace.stage_busy[slot]:
                                workspace.stage_free[slot].synchronize()
                                workspace.stage_busy[slot] = False

                        def consume(loaded: _LoadedPage, stage: _KVStage) -> None:
                            nonlocal initialize, kv_tile_index
                            page = loaded.page
                            for kv_offset in range(0, page.valid_tokens, plan.kv_chunk_tokens):
                                kv_tokens = min(
                                    plan.kv_chunk_tokens, page.valid_tokens - kv_offset
                                )
                                buffer_index = kv_tile_index % len(workspace.k)
                                with torch.cuda.stream(workspace.h2d_stream):
                                    if workspace.kv_busy[buffer_index]:
                                        workspace.h2d_stream.wait_event(
                                            workspace.kv_free[buffer_index]
                                        )
                                    copy_started = time.perf_counter()
                                    workspace.k[buffer_index][:kv_tokens].copy_(
                                        stage.k[kv_offset : kv_offset + kv_tokens],
                                        non_blocking=True,
                                    )
                                    workspace.v[buffer_index][:kv_tokens].copy_(
                                        stage.v[kv_offset : kv_offset + kv_tokens],
                                        non_blocking=True,
                                    )
                                    scale_groups = 0
                                    if kv_layout.storage_dtype == "int8":
                                        assert stage.k_scales is not None
                                        assert stage.v_scales is not None
                                        assert workspace.k_scales is not None
                                        assert workspace.v_scales is not None
                                        scale_start = kv_offset // kv_layout.quant_group_tokens
                                        scale_token_offset = (
                                            kv_offset % kv_layout.quant_group_tokens
                                        )
                                        scale_groups = math.ceil(
                                            (scale_token_offset + kv_tokens)
                                            / kv_layout.quant_group_tokens
                                        )
                                        workspace.k_scales[buffer_index][:scale_groups].copy_(
                                            stage.k_scales[
                                                scale_start : scale_start + scale_groups
                                            ],
                                            non_blocking=True,
                                        )
                                        workspace.v_scales[buffer_index][:scale_groups].copy_(
                                            stage.v_scales[
                                                scale_start : scale_start + scale_groups
                                            ],
                                            non_blocking=True,
                                        )
                                    stats.pinned_copy_seconds += (
                                        time.perf_counter() - copy_started
                                    )
                                    workspace.kv_ready[buffer_index].record(
                                        workspace.h2d_stream
                                    )
                                storage_bytes = (
                                    2
                                    * kv_tokens
                                    * plan.kv_heads
                                    * plan.head_dim
                                    * stage.k.element_size()
                                    + 2 * scale_groups * plan.kv_heads * 2
                                )
                                stats.h2d_bytes += storage_bytes
                                with torch.cuda.stream(workspace.compute_stream):
                                    workspace.compute_stream.wait_event(
                                        workspace.kv_ready[buffer_index]
                                    )
                                    if kv_layout.storage_dtype == "int8":
                                        assert workspace.k_scales is not None
                                        assert workspace.v_scales is not None
                                        update_attention_state_int8(
                                            workspace.q,
                                            workspace.k[buffer_index],
                                            workspace.v[buffer_index],
                                            workspace.k_scales[buffer_index],
                                            workspace.v_scales[buffer_index],
                                            workspace.running_max,
                                            workspace.running_sum,
                                            workspace.accumulator,
                                            q_tokens=q_tokens,
                                            kv_tokens=kv_tokens,
                                            q_local_offset=q_local_offset,
                                            kv_local_offset=(
                                                page.segment_token_start + kv_offset
                                            ),
                                            storage_token_offset=scale_token_offset,
                                            causal_shift=causal_shift,
                                            softmax_scale=scale,
                                            causal=causal,
                                            initialize=initialize,
                                            block_m=plan.block_m,
                                            block_n=plan.block_n,
                                            num_warps=plan.num_warps,
                                            num_stages=plan.num_stages,
                                            quant_group_tokens=kv_layout.quant_group_tokens,
                                        )
                                    else:
                                        update_attention_state(
                                            workspace.q,
                                            workspace.k[buffer_index],
                                            workspace.v[buffer_index],
                                            workspace.running_max,
                                            workspace.running_sum,
                                            workspace.accumulator,
                                            q_tokens=q_tokens,
                                            kv_tokens=kv_tokens,
                                            q_local_offset=q_local_offset,
                                            kv_local_offset=(
                                                page.segment_token_start + kv_offset
                                            ),
                                            causal_shift=causal_shift,
                                            softmax_scale=scale,
                                            causal=causal,
                                            initialize=initialize,
                                            block_m=plan.block_m,
                                            block_n=plan.block_n,
                                            num_warps=plan.num_warps,
                                            num_stages=plan.num_stages,
                                        )
                                    workspace.kv_free[buffer_index].record(
                                        workspace.compute_stream
                                    )
                                workspace.kv_busy[buffer_index] = True
                                initialize = False
                                kv_tile_index += 1
                                stats.state_spills += 1
                            with torch.cuda.stream(workspace.h2d_stream):
                                workspace.stage_free[loaded.stage_index].record(
                                    workspace.h2d_stream
                                )
                            workspace.stage_busy[loaded.stage_index] = True

                        self._scan_kv_pages(
                            kv_pages,
                            staging.kv,
                            kv_reader,
                            executor,
                            cache,
                            consume,
                            stats,
                            source_is_nvme=kv_is_nvme,
                            stage_reuse_wait=wait_stage,
                        )
                        with torch.cuda.stream(workspace.compute_stream):
                            workspace.q_free.record(workspace.compute_stream)
                            workspace.q_busy = True
                            if workspace.output_busy[output_index]:
                                workspace.compute_stream.wait_event(
                                    workspace.output_free[output_index]
                                )
                            finalize_attention(
                                workspace.accumulator,
                                workspace.running_sum,
                                workspace.output[output_index],
                                q_tokens=q_tokens,
                            )
                            workspace.output_ready[output_index].record(
                                workspace.compute_stream
                            )
                        with torch.cuda.stream(workspace.d2h_stream):
                            workspace.d2h_stream.wait_event(
                                workspace.output_ready[output_index]
                            )
                            workspace.output[output_index][:q_tokens].record_stream(
                                workspace.d2h_stream
                            )
                            staging.outputs[output_host_slot][
                                page_offset : page_offset + q_tokens
                            ].copy_(
                                workspace.output[output_index][:q_tokens],
                                non_blocking=True,
                            )
                            workspace.output_free[output_index].record(
                                workspace.d2h_stream
                            )
                            workspace.output_host_ready[output_host_slot].record(
                                workspace.d2h_stream
                            )
                        workspace.output_busy[output_index] = True
                        stats.d2h_bytes += q_tokens * q_layout_bytes(plan)
                        q_chunk_index += 1
                    workspace.output_host_ready[output_host_slot].synchronize()
                    self._submit_output(
                        q_page,
                        staging.outputs[output_host_slot],
                        output_host_slot,
                        writer,
                        executor,
                        output_futures,
                        stats,
                        output_is_nvme,
                    )
            if kernel_started:
                with torch.cuda.stream(workspace.compute_stream):
                    kernel_end.record(workspace.compute_stream)
                kernel_end.synchronize()
                stats.gpu_kernel_seconds += kernel_start.elapsed_time(kernel_end) / 1000.0
            workspace.d2h_stream.synchronize()


def q_layout_bytes(plan: AttentionPlan) -> int:
    return plan.q_heads * plan.head_dim * torch.empty((), dtype=plan.dtype).element_size()


__all__ = ["PagedAttentionRunner"]
