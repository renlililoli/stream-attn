from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch

from ..kernels import finalize_attention, update_attention_state
from ..stats import StreamingAttentionStats
from .dynamic import QueryTaskMeasurement
from .tasks import QueryTask
from .tile_source import HostQKVTileSource, QKVTileSource


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
        task_measurement: QueryTaskMeasurement | None = None,
        task_lifecycle: bool = False,
    ) -> torch.Tensor | None:
        workspace = self._workspace
        assert workspace is not None
        source = HostQKVTileSource(
            q_cpu,
            k_cpu,
            v_cpu,
            workspace,
            enable_nvtx=self.config.enable_nvtx,
        )
        return self._run_triton_from_source(
            source,
            query_tasks,
            scale,
            causal,
            out_cpu,
            stats,
            output_transform=output_transform,
            output_consumer=output_consumer,
            task_measurement=task_measurement,
            task_lifecycle=task_lifecycle,
        )

    def _run_triton_from_source(
        self,
        source: QKVTileSource,
        query_tasks: tuple[QueryTask, ...],
        scale: float,
        causal: bool,
        out_cpu: torch.Tensor | None,
        stats: StreamingAttentionStats,
        output_transform: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
        output_consumer=None,
        task_measurement: QueryTaskMeasurement | None = None,
        task_lifecycle: bool = False,
    ) -> torch.Tensor | None:
        if output_transform is not None and output_consumer is not None:
            raise ValueError("output_transform and output_consumer are mutually exclusive")
        if output_consumer is None and out_cpu is None:
            raise ValueError("out_cpu is required without an output_consumer")
        if task_measurement is not None and len(query_tasks) != 1:
            raise ValueError("task timing requires exactly one query task")
        if task_lifecycle and output_consumer is None:
            raise ValueError("task_lifecycle requires an output_consumer")
        workspace = self._workspace
        assert workspace is not None
        plan = self.plan
        compute_stream = workspace.compute_stream
        q_chunk_index = 0
        task_done_event = None
        h2d_bytes_before = stats.h2d_bytes
        d2h_bytes_before = stats.d2h_bytes

        with torch.cuda.device(plan.device):
            timing = workspace.get_task_timing() if task_measurement is not None else None
            workspace.pipeline_start.record(compute_stream)
            for task in query_tasks:
                if task_lifecycle:
                    output_consumer.begin_task(task)
                q_tile_start = task.q_start
                q_tile_stop = task.q_stop
                q_tokens = task.q_tokens
                output_index = q_chunk_index % plan.num_output_buffers
                reuse_q_for_output = (
                    output_transform is not None or output_consumer is not None
                ) and plan.output_mode == "device_consumer"

                if timing is not None:
                    with torch.cuda.stream(workspace.h2d_stream):
                        timing.task_start.record(workspace.h2d_stream)
                        timing.h2d_start.record(workspace.h2d_stream)
                stats.q_chunks += 1
                stats.max_resident_q_tokens = max(stats.max_resident_q_tokens, q_tokens)
                with torch.cuda.stream(compute_stream):
                    source.load_q(
                        workspace.q[:q_tokens],
                        q_tile_start,
                        q_tile_stop,
                        compute_stream,
                        stats,
                    )
                    if timing is not None:
                        timing.attention_start.record(compute_stream)

                    initialize = True
                    for kv_tile_index, kv_tile_start in enumerate(
                        range(task.k_start, task.k_stop, plan.kv_chunk_tokens)
                    ):
                        kv_tile_stop = min(kv_tile_start + plan.kv_chunk_tokens, task.k_stop)
                        kv_tokens = kv_tile_stop - kv_tile_start
                        buffer_index = kv_tile_index % plan.num_kv_buffers
                        source.load_kv(
                            workspace.k[buffer_index][:kv_tokens],
                            workspace.v[buffer_index][:kv_tokens],
                            buffer_index,
                            kv_tile_start,
                            kv_tile_stop,
                            compute_stream,
                            stats,
                        )
                        stats.kv_tiles += 1
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
                        source.release_kv(buffer_index, compute_stream)

                    if timing is not None:
                        with torch.cuda.stream(workspace.h2d_stream):
                            timing.h2d_end.record(workspace.h2d_stream)

                    if not reuse_q_for_output:
                        source.release_q(compute_stream)
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
                    if timing is not None:
                        timing.attention_end.record(compute_stream)
                    output_gpu = finalize_output[:q_tokens]
                    output_aliases_q = reuse_q_for_output
                    if output_consumer is not None:
                        if timing is not None:
                            timing.consumer_start.record(compute_stream)
                        with self._range("seqattn:device_output_consumer"):
                            output_consumer(
                                output_gpu.reshape(q_tokens, -1),
                                q_tile_start,
                                q_tile_stop,
                            )
                        if task_lifecycle:
                            task_done_event = output_consumer.finish_task()
                        if timing is not None:
                            timing.consumer_end.record(compute_stream)
                        source.release_q(compute_stream)
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
                                f"{tuple(output_gpu.shape)} != {tuple(output_slice_shape)}"
                            )
                        if output_gpu.dtype != out_cpu.dtype:
                            raise ValueError("output_transform result dtype must match out dtype")
                        output_aliases_q = (
                            output_gpu.untyped_storage().data_ptr()
                            == workspace.q.untyped_storage().data_ptr()
                        )
                        if reuse_q_for_output and not output_aliases_q:
                            source.release_q(compute_stream)
                    if output_consumer is None:
                        workspace.output_ready[output_index].record(compute_stream)
                if output_consumer is None:
                    assert out_cpu is not None
                    with self._range("seqattn:output_d2h"), torch.cuda.stream(workspace.d2h_stream):
                        if timing is not None:
                            timing.d2h_start.record(workspace.d2h_stream)
                        workspace.d2h_stream.wait_event(workspace.output_ready[output_index])
                        out_cpu[q_tile_start:q_tile_stop].copy_(
                            output_gpu, non_blocking=out_cpu.is_pinned()
                        )
                        output_gpu.record_stream(workspace.d2h_stream)
                        workspace.output_free[output_index].record(workspace.d2h_stream)
                        if reuse_q_for_output and output_aliases_q:
                            source.release_q(workspace.d2h_stream)
                        if timing is not None:
                            timing.d2h_end.record(workspace.d2h_stream)
                            timing.task_done.record(workspace.d2h_stream)
                            task_done_event = timing.task_done
                    workspace.output_has_pending_copy[output_index] = True
                    stats.d2h_bytes += output_gpu.numel() * output_gpu.element_size()
                q_chunk_index += 1
            if output_consumer is not None and not task_lifecycle:
                with torch.cuda.stream(compute_stream):
                    output_consumer.finish()
            workspace.pipeline_end.record(compute_stream)
            if output_consumer is None:
                workspace.d2h_stream.synchronize()
            elif task_lifecycle:
                if task_done_event is None:
                    raise RuntimeError("task consumer did not return a completion event")
                task_done_event.synchronize()
            else:
                output_consumer.synchronize()
            workspace.pipeline_end.synchronize()
            stats.compute_pipeline_seconds += (
                workspace.pipeline_start.elapsed_time(workspace.pipeline_end) / 1000.0
            )
            if task_measurement is not None:
                assert timing is not None
                assert task_done_event is not None
                task = query_tasks[0]
                task_measurement.h2d_seconds = (
                    timing.h2d_start.elapsed_time(timing.h2d_end) / 1000.0
                )
                task_measurement.attention_seconds = (
                    timing.attention_start.elapsed_time(timing.attention_end) / 1000.0
                )
                if output_consumer is not None:
                    task_measurement.consumer_seconds = (
                        timing.consumer_start.elapsed_time(timing.consumer_end) / 1000.0
                    )
                    d2h_seconds = getattr(output_consumer, "task_d2h_seconds", None)
                    if d2h_seconds is not None:
                        task_measurement.d2h_seconds = float(d2h_seconds())
                    d2h_bytes = getattr(output_consumer, "task_d2h_bytes", None)
                    if d2h_bytes is not None:
                        task_measurement.d2h_bytes = int(d2h_bytes())
                else:
                    task_measurement.d2h_seconds = (
                        timing.d2h_start.elapsed_time(timing.d2h_end) / 1000.0
                    )
                task_measurement.elapsed_seconds = (
                    timing.task_start.elapsed_time(task_done_event) / 1000.0
                )
                task_measurement.h2d_bytes = stats.h2d_bytes - h2d_bytes_before
                if task_measurement.d2h_bytes <= 0:
                    task_measurement.d2h_bytes = stats.d2h_bytes - d2h_bytes_before
                task_measurement.attention_flops = (
                    4 * task.q_tokens * task.k_tokens * plan.q_heads * plan.head_dim
                )
        return out_cpu


__all__ = ["TritonExecutorMixin"]
