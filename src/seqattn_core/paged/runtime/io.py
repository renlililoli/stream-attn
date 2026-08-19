from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ...stats import PagedAttentionStats
from ..cache import CacheLookup, KVPageCache
from ..layout import PageDescriptor, PageReadMetrics
from ..memory_budget import HostMemorySnapshot
from ..protocols import PageReader, PageWriter
from .types import KVStage, LoadedPage


def load_kv_page(
    reader: PageReader,
    cache: KVPageCache,
    page: PageDescriptor,
    stage: KVStage,
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


class PagedIoMixin:
    @staticmethod
    def _record_memory_stats(stats: PagedAttentionStats, snapshot: HostMemorySnapshot) -> None:
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
        stats.simulated_read_seconds += metrics.simulated_io_seconds
        stats.simulated_read_service_seconds += metrics.simulated_service_seconds
        stats.simulated_read_queue_seconds += metrics.simulated_queue_seconds
        stats.simulated_logical_read_bytes += metrics.simulated_logical_bytes
        stats.simulated_physical_read_bytes += metrics.simulated_physical_bytes

    @staticmethod
    def _accumulate_write(
        stats: PagedAttentionStats,
        metrics: PageReadMetrics,
        output_is_nvme: bool,
    ) -> None:
        stats.output_write_seconds += metrics.read_seconds
        if output_is_nvme:
            stats.nvme_write_seconds += metrics.read_seconds
            stats.nvme_logical_write_bytes += metrics.logical_bytes
            stats.nvme_physical_write_bytes += metrics.physical_bytes
        stats.simulated_write_seconds += metrics.simulated_io_seconds
        stats.simulated_write_service_seconds += metrics.simulated_service_seconds
        stats.simulated_write_queue_seconds += metrics.simulated_queue_seconds
        stats.simulated_logical_write_bytes += metrics.simulated_logical_bytes
        stats.simulated_physical_write_bytes += metrics.simulated_physical_bytes

    def _finish_output_futures(
        self,
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
            self._accumulate_write(stats, metrics, output_is_nvme)

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
            self._accumulate_write(stats, metrics, output_is_nvme)
        futures[slot] = executor.submit(writer.write_page, page, data[: page.valid_tokens])

    def _scan_kv_pages(
        self,
        pages: Sequence[PageDescriptor],
        stages: Sequence[KVStage],
        reader: PageReader,
        executor: ThreadPoolExecutor,
        cache: KVPageCache,
        consume: Callable[[LoadedPage, KVStage], None],
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
                load_kv_page,
                reader,
                cache,
                pages[index],
                stages[slot],
            )

        for index in range(depth):
            submit(index)
        for index, page in enumerate(pages):
            with self._range("seqattn_paged:kv_page_wait"):
                wait_started = time.perf_counter()
                metrics, lookup = pending.pop(index).result()
                stats.io_queue_wait_seconds += time.perf_counter() - wait_started
            self._accumulate_read(stats, metrics, lookup, source_is_nvme)
            slot = index % len(stages)
            loaded = LoadedPage(page, slot, metrics, lookup)
            consume(loaded, stages[slot])
            stats.kv_pages += 1
            next_index = index + depth
            if next_index < len(pages):
                submit(next_index)


__all__ = ["PagedIoMixin"]
