from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ...kernels import (
    finalize_attention,
    update_attention_state,
    update_attention_state_int8,
)
from ...planner import AttentionPlan
from ...stats import PagedAttentionStats
from ..cache import KVPageCache
from ..layout import KVLayout, PageDescriptor, PageReadMetrics
from ..protocols import PageReader, PageWriter
from .staging import HostStaging, PagedCudaWorkspace
from .types import KVStage, LoadedPage


def q_layout_bytes(plan: AttentionPlan) -> int:
    return plan.q_heads * plan.head_dim * torch.empty((), dtype=plan.dtype).element_size()


class TritonExecutorMixin:
    def _run_triton(
        self,
        plan: AttentionPlan,
        kv_layout: KVLayout,
        q_reader: PageReader,
        kv_reader: PageReader,
        writer: PageWriter,
        executor: ThreadPoolExecutor,
        cache: KVPageCache,
        staging: HostStaging,
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
        workspace = PagedCudaWorkspace(plan, kv_layout)
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
                        self._accumulate_write(stats, metrics, output_is_nvme)
                        output_futures[output_host_slot] = None
                    with self._range("seqattn_paged:q_read"):
                        metrics = q_reader.read_q(q_page, staging.q)
                    self._accumulate_read(stats, metrics, None, q_is_nvme)
                    stats.q_pages += 1
                    for page_offset in range(0, q_page.valid_tokens, plan.q_chunk_tokens):
                        q_tokens = min(plan.q_chunk_tokens, q_page.valid_tokens - page_offset)
                        q_local_offset = q_page.segment_token_start + page_offset
                        output_index = q_chunk_index % len(workspace.output)
                        with (
                            self._range("seqattn_paged:q_h2d"),
                            torch.cuda.stream(workspace.h2d_stream),
                        ):
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
                                with self._range("seqattn_paged:stage_reuse_wait"):
                                    workspace.stage_free[slot].synchronize()
                                workspace.stage_busy[slot] = False

                        def consume(
                            loaded: LoadedPage,
                            stage: KVStage,
                            *,
                            q_tokens: int = q_tokens,
                            q_local_offset: int = q_local_offset,
                            causal_shift: int = causal_shift,
                        ) -> None:
                            nonlocal initialize, kv_tile_index
                            page = loaded.page
                            for kv_offset in range(0, page.valid_tokens, plan.kv_chunk_tokens):
                                kv_tokens = min(plan.kv_chunk_tokens, page.valid_tokens - kv_offset)
                                buffer_index = kv_tile_index % len(workspace.k)
                                with (
                                    self._range("seqattn_paged:kv_h2d"),
                                    torch.cuda.stream(workspace.h2d_stream),
                                ):
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
                                    scale_token_offset = 0
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
                                    stats.pinned_copy_seconds += time.perf_counter() - copy_started
                                    workspace.kv_ready[buffer_index].record(workspace.h2d_stream)
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
                                    with self._range("seqattn_paged:fused_update"):
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
                                                quant_group_tokens=(kv_layout.quant_group_tokens),
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
                                    workspace.kv_free[buffer_index].record(workspace.compute_stream)
                                workspace.kv_busy[buffer_index] = True
                                initialize = False
                                kv_tile_index += 1
                                stats.state_spills += 1
                            with torch.cuda.stream(workspace.h2d_stream):
                                workspace.stage_free[loaded.stage_index].record(
                                    workspace.h2d_stream
                                )
                            workspace.stage_busy[loaded.stage_index] = True

                        with self._range("seqattn_paged:kv_scan"):
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
                            with self._range("seqattn_paged:fused_finalize"):
                                finalize_attention(
                                    workspace.accumulator,
                                    workspace.running_sum,
                                    workspace.output[output_index],
                                    q_tokens=q_tokens,
                                )
                            workspace.output_ready[output_index].record(workspace.compute_stream)
                        with (
                            self._range("seqattn_paged:output_d2h"),
                            torch.cuda.stream(workspace.d2h_stream),
                        ):
                            workspace.d2h_stream.wait_event(workspace.output_ready[output_index])
                            workspace.output[output_index][:q_tokens].record_stream(
                                workspace.d2h_stream
                            )
                            staging.outputs[output_host_slot][
                                page_offset : page_offset + q_tokens
                            ].copy_(
                                workspace.output[output_index][:q_tokens],
                                non_blocking=True,
                            )
                            workspace.output_free[output_index].record(workspace.d2h_stream)
                            workspace.output_host_ready[output_host_slot].record(
                                workspace.d2h_stream
                            )
                        workspace.output_busy[output_index] = True
                        stats.d2h_bytes += q_tokens * q_layout_bytes(plan)
                        q_chunk_index += 1
                    with self._range("seqattn_paged:output_wait"):
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


__all__ = ["TritonExecutorMixin"]
