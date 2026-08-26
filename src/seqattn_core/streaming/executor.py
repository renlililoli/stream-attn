from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch

from ..kernels import finalize_attention, update_attention_state
from ..stats import StreamingAttentionStats
from .tasks import QueryTask


class TritonExecutorMixin:
    def _range(self, name: str):
        if self.config.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _run_triton(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        query_tasks: tuple[QueryTask, ...],
        scale: float,
        causal: bool,
        out_cpu: torch.Tensor | None,
        stats: StreamingAttentionStats,
        output_transform: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
        output_consumer=None,
    ) -> torch.Tensor | None:
        if output_transform is not None and output_consumer is not None:
            raise ValueError("output_transform and output_consumer are mutually exclusive")
        if output_consumer is None and out_cpu is None:
            raise ValueError("out_cpu is required without an output_consumer")
        workspace = self._workspace
        assert workspace is not None
        plan = self.plan
        compute_stream = workspace.compute_stream
        q_chunk_index = 0

        with torch.cuda.device(plan.device):
            workspace.pipeline_start.record(compute_stream)
            for task in query_tasks:
                q_tile_start = task.q_start
                q_tile_stop = task.q_stop
                q_tokens = task.q_tokens
                output_index = q_chunk_index % plan.num_output_buffers
                reuse_q_for_output = (
                    output_transform is not None or output_consumer is not None
                ) and plan.output_mode == "device_consumer"

                with self._range("seqattn:q_h2d"), torch.cuda.stream(workspace.h2d_stream):
                    if workspace.q_has_pending_compute:
                        workspace.h2d_stream.wait_event(workspace.q_free)
                    workspace.q[:q_tokens].copy_(
                        q_cpu[q_tile_start:q_tile_stop],
                        non_blocking=q_cpu.is_pinned(),
                    )
                    workspace.q_ready.record(workspace.h2d_stream)
                stats.h2d_bytes += q_tokens * q_cpu.shape[1] * q_cpu.shape[2] * q_cpu.element_size()
                stats.q_chunks += 1
                stats.max_resident_q_tokens = max(stats.max_resident_q_tokens, q_tokens)
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(workspace.q_ready)

                    initialize = True
                    for kv_tile_index, kv_tile_start in enumerate(
                        range(task.k_start, task.k_stop, plan.kv_chunk_tokens)
                    ):
                        kv_tile_stop = min(kv_tile_start + plan.kv_chunk_tokens, task.k_stop)
                        kv_tokens = kv_tile_stop - kv_tile_start
                        buffer_index = kv_tile_index % plan.num_kv_buffers
                        with (
                            self._range("seqattn:kv_h2d"),
                            torch.cuda.stream(workspace.h2d_stream),
                        ):
                            if workspace.kv_has_pending_compute[buffer_index]:
                                workspace.h2d_stream.wait_event(workspace.kv_free[buffer_index])
                            workspace.k[buffer_index][:kv_tokens].copy_(
                                k_cpu[kv_tile_start:kv_tile_stop],
                                non_blocking=k_cpu.is_pinned(),
                            )
                            workspace.v[buffer_index][:kv_tokens].copy_(
                                v_cpu[kv_tile_start:kv_tile_stop],
                                non_blocking=v_cpu.is_pinned(),
                            )
                            workspace.kv_ready[buffer_index].record(workspace.h2d_stream)
                        stats.h2d_bytes += (
                            2 * kv_tokens * k_cpu.shape[1] * k_cpu.shape[2] * k_cpu.element_size()
                        )
                        stats.kv_tiles += 1
                        compute_stream.wait_event(workspace.kv_ready[buffer_index])
                        with self._range("seqattn:fused_update"):
                            update_attention_state(
                                workspace.q,
                                workspace.k[buffer_index],
                                workspace.v[buffer_index],
                                workspace.running_max,
                                workspace.running_sum,
                                workspace.accumulator,
                                q_tokens=q_tokens,
                                kv_tokens=kv_tokens,
                                q_local_offset=task.q_local_offset,
                                kv_local_offset=kv_tile_start - task.k_start,
                                causal_shift=task.causal_shift,
                                softmax_scale=scale,
                                causal=causal,
                                initialize=initialize,
                                block_m=plan.block_m,
                                block_n=plan.block_n,
                                num_warps=plan.num_warps,
                                num_stages=plan.num_stages,
                            )
                        initialize = False
                        workspace.kv_free[buffer_index].record(compute_stream)
                        workspace.kv_has_pending_compute[buffer_index] = True

                    if not reuse_q_for_output:
                        workspace.q_free.record(compute_stream)
                        workspace.q_has_pending_compute = True
                    if (
                        output_consumer is None
                        and not reuse_q_for_output
                        and workspace.output_has_pending_copy[output_index]
                    ):
                        compute_stream.wait_event(workspace.output_free[output_index])
                    finalize_output = (
                        workspace.q if reuse_q_for_output else workspace.output[output_index]
                    )
                    with self._range("seqattn:fused_finalize"):
                        finalize_attention(
                            workspace.accumulator,
                            workspace.running_sum,
                            finalize_output,
                            q_tokens=q_tokens,
                        )
                    output_gpu = finalize_output[:q_tokens]
                    output_aliases_q = reuse_q_for_output
                    if output_consumer is not None:
                        with self._range("seqattn:device_output_consumer"):
                            output_consumer(
                                output_gpu.reshape(q_tokens, -1),
                                q_tile_start,
                                q_tile_stop,
                            )
                        workspace.q_free.record(compute_stream)
                        workspace.q_has_pending_compute = True
                    elif output_transform is not None:
                        assert out_cpu is not None
                        with self._range("seqattn:device_output_transform"):
                            output_gpu = output_transform(
                                output_gpu.reshape(q_tokens, -1),
                                q_tile_start,
                                q_tile_stop,
                            )
                        if output_gpu.device.type != "cuda":
                            raise ValueError("output_transform must return a CUDA tensor")
                        output_slice_shape = out_cpu[q_tile_start:q_tile_stop].shape
                        if output_gpu.shape != output_slice_shape:
                            raise ValueError(
                                "output_transform result shape does not match "
                                "the output slice: "
                                f"{tuple(output_gpu.shape)} != "
                                f"{tuple(output_slice_shape)}"
                            )
                        if output_gpu.dtype != out_cpu.dtype:
                            raise ValueError("output_transform result dtype must match out dtype")
                        output_aliases_q = (
                            output_gpu.untyped_storage().data_ptr()
                            == workspace.q.untyped_storage().data_ptr()
                        )
                        if reuse_q_for_output and not output_aliases_q:
                            workspace.q_free.record(compute_stream)
                            workspace.q_has_pending_compute = True
                    if output_consumer is None:
                        workspace.output_ready[output_index].record(compute_stream)
                if output_consumer is None:
                    assert out_cpu is not None
                    with self._range("seqattn:output_d2h"), torch.cuda.stream(workspace.d2h_stream):
                        workspace.d2h_stream.wait_event(workspace.output_ready[output_index])
                        out_cpu[q_tile_start:q_tile_stop].copy_(
                            output_gpu, non_blocking=out_cpu.is_pinned()
                        )
                        output_gpu.record_stream(workspace.d2h_stream)
                        workspace.output_free[output_index].record(workspace.d2h_stream)
                        if reuse_q_for_output and output_aliases_q:
                            workspace.q_free.record(workspace.d2h_stream)
                            workspace.q_has_pending_compute = True
                    workspace.output_has_pending_copy[output_index] = True
                    stats.d2h_bytes += output_gpu.numel() * output_gpu.element_size()
                q_chunk_index += 1
            if output_consumer is not None:
                with torch.cuda.stream(compute_stream):
                    output_consumer.finish()
            workspace.pipeline_end.record(compute_stream)
            if output_consumer is None:
                workspace.d2h_stream.synchronize()
            else:
                output_consumer.synchronize()
            workspace.pipeline_end.synchronize()
            stats.compute_pipeline_seconds += (
                workspace.pipeline_start.elapsed_time(workspace.pipeline_end) / 1000.0
            )
        return out_cpu


__all__ = ["TritonExecutorMixin"]
