from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ...config import PagedAttentionConfig
from ...planner import build_plan
from ...stats import PagedAttentionStats
from ...streaming.backend import resolve_backend
from ..cache import KVPageCache
from ..layout import PageReadMetrics, pages_by_segment
from ..memory_budget import HostMemoryPlan, HostMemorySnapshot
from ..protocols import PageReader, PageSink, PageSource, PageWriter
from .contracts import backing_is_nvme, validate_source_contract
from .io import PagedIoMixin
from .reference import ReferenceExecutorMixin
from .staging import HostStaging
from .triton import TritonExecutorMixin


class PagedAttentionRunner(PagedIoMixin, ReferenceExecutorMixin, TritonExecutorMixin):
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
            raise ValueError("NVMe output sink direct_io mode must match PagedAttentionConfig")
        q_layout, kv_layout, q_bounds, k_bounds = validate_source_contract(
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
        backend = resolve_backend(
            attention_config.backend, plan.dtype, plan.device, head_dim=plan.head_dim
        )
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
        staging = HostStaging(
            q_layout,
            kv_layout,
            q_source.q_pages,
            kv_source.kv_pages,
            queue_depth=config.io_queue_depth,
            output_buffers=config.num_output_buffers,
            pinned=backend == "triton",
            memory_plan=memory_plan,
        )
        q_reader: PageReader | None = None
        kv_reader: PageReader | None = None
        writer: PageWriter | None = None
        cache: KVPageCache | None = None
        executor: ThreadPoolExecutor | None = None
        output_futures: list[Future[PageReadMetrics] | None] = [None for _ in staging.outputs]
        output_is_nvme = getattr(output_sink, "backing_kind", "memory") == "nvme"
        completed_snapshot: HostMemorySnapshot | None = None
        started = time.perf_counter()
        try:
            if q_source is kv_source:
                q_reader = kv_reader = q_source.open_reader(memory_plan, config.io_queue_depth)
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
            common = (
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
                backing_is_nvme(q_source),
                backing_is_nvme(kv_source),
                output_is_nvme,
            )
            if backend == "reference":
                self._run_reference(*common)
            else:
                self._run_triton(plan, kv_layout, *common[1:])
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


__all__ = ["PagedAttentionRunner"]
