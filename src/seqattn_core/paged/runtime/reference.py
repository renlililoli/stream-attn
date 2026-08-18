from __future__ import annotations

import math
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ...planner import AttentionPlan
from ...quantization import dequantize_int8_per_token_group
from ...stats import PagedAttentionStats
from ..cache import KVPageCache
from ..layout import PageDescriptor, PageReadMetrics
from ..protocols import PageReader, PageWriter
from .staging import HostStaging
from .types import KVStage, LoadedPage


class ReferenceExecutorMixin:
    def _run_reference(
        self,
        plan: AttentionPlan,
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

                    def consume(
                        loaded: LoadedPage,
                        stage: KVStage,
                        *,
                        q: torch.Tensor = q,
                        q_positions: torch.Tensor = q_positions,
                        causal_shift: int = causal_shift,
                        running_max: torch.Tensor = running_max,
                        running_sum: torch.Tensor = running_sum,
                        accumulator: torch.Tensor = accumulator,
                    ) -> None:
                        page = loaded.page
                        if plan.dtype == torch.int8:
                            raise AssertionError("query dtype cannot be int8")
                        if stage.k.dtype == torch.int8:
                            assert stage.k_scales is not None and stage.v_scales is not None
                            groups = math.ceil(page.valid_tokens / cache.layout.quant_group_tokens)
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


__all__ = ["ReferenceExecutorMixin"]
