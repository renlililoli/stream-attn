from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import torch
from seqattn_core._plugin_api import (
    DeviceOutputConsumer,
    QueryTask,
    QueryTaskMeasurement,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    TaskDeviceOutputConsumer,
    validate_host_qkv,
)

from .dynamic import (
    DynamicQController,
    DynamicQueryCursor,
    DynamicScheduleConfig,
    DynamicWorkloadSignature,
)
from .stats import DynamicDeviceStats, DynamicTaskTrace, MultiGpuAttentionStats

if TYPE_CHECKING:
    from .planning import DeviceQuerySchedule, MultiGpuAttentionPlan


class MultiGpuStreamingAttentionRunner:
    """Single-flight executor for static or completion-driven multi-GPU Q scheduling."""

    def __init__(
        self,
        plan: MultiGpuAttentionPlan,
        *,
        runner_overrides: dict[torch.device | str, StreamingAttentionRunner] | None = None,
        schedule_mode: Literal["static", "dynamic"] | None = None,
        dynamic_config: DynamicScheduleConfig | None = None,
    ) -> None:
        self.plan = plan
        self.schedule_mode = plan.schedule_mode if schedule_mode is None else schedule_mode
        if self.schedule_mode not in {"static", "dynamic"}:
            raise ValueError("schedule_mode must be 'static' or 'dynamic'")
        if self.schedule_mode == "static" and plan.schedule_mode == "dynamic":
            raise ValueError("a dynamic plan cannot be executed as a static schedule")
        self.dynamic_config = plan.dynamic_config if dynamic_config is None else dynamic_config
        if self.schedule_mode == "dynamic" and self.dynamic_config is None:
            self.dynamic_config = DynamicScheduleConfig()
        if self.dynamic_config is not None:
            self.dynamic_config.validate()
        overrides = {
            str(torch.device(device)): runner
            for device, runner in ({} if runner_overrides is None else runner_overrides).items()
        }
        expected = {str(device) for device in plan.devices}
        if not set(overrides) <= expected:
            raise ValueError("runner_overrides contains a device outside the multi-GPU plan")
        runners = []
        for schedule in plan.schedules:
            runner = overrides.get(str(schedule.device))
            if runner is None:
                runner = StreamingAttentionRunner(schedule.attention_plan)
            elif runner.plan != schedule.attention_plan:
                raise ValueError(f"runner override plan does not match {schedule.device}")
            runners.append(runner)
        self.runners = tuple(runners)
        element_size = torch.empty((), dtype=plan.dtype).element_size()
        self.controllers = tuple(
            DynamicQController(
                initial_q_tokens=schedule.initial_q_tokens,
                q_min_tokens=schedule.q_min_tokens,
                q_capacity_tokens=schedule.q_capacity_tokens,
                block_m=schedule.attention_plan.block_m,
                compute_tflops=schedule.device_spec.compute_tflops,
                h2d_gbps=schedule.device_spec.h2d_gbps,
                d2h_gbps=(
                    schedule.device_spec.h2d_gbps
                    if schedule.device_spec.d2h_gbps is None
                    else schedule.device_spec.d2h_gbps
                ),
                q_heads=plan.q_heads,
                kv_heads=plan.kv_heads,
                element_size=element_size,
                config=self.dynamic_config or DynamicScheduleConfig(),
            )
            for schedule in plan.schedules
        )
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.runners),
            thread_name_prefix="seqattn-gpu",
        )
        self._run_lock = threading.Lock()
        self._closed = False

    def _workload_signature(
        self,
        schedule: DeviceQuerySchedule,
        consumer_mode: str,
    ) -> DynamicWorkloadSignature:
        attention_plan = schedule.attention_plan
        return DynamicWorkloadSignature(
            q_segment_lengths=tuple(stop - start for start, stop in pairwise(self.plan.q_bounds)),
            k_segment_lengths=tuple(stop - start for start, stop in pairwise(self.plan.k_bounds)),
            q_heads=self.plan.q_heads,
            kv_heads=self.plan.kv_heads,
            head_dim=self.plan.head_dim,
            dtype=str(self.plan.dtype),
            kernel_profile=(
                attention_plan.block_m,
                attention_plan.block_n,
                attention_plan.num_warps,
                attention_plan.num_stages,
            ),
            consumer_mode=consumer_mode,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _validate_runtime_inputs(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> None:
        q_bounds, k_bounds = validate_host_qkv(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        if tuple(q_bounds) != self.plan.q_bounds or tuple(k_bounds) != self.plan.k_bounds:
            raise ValueError("runtime cu_seqlens do not match the multi-GPU plan")
        if q_cpu.shape != (
            self.plan.max_q_tokens,
            self.plan.q_heads,
            self.plan.head_dim,
        ):
            raise ValueError("q shape does not match the multi-GPU plan")
        if k_cpu.shape != (
            self.plan.max_kv_tokens,
            self.plan.kv_heads,
            self.plan.head_dim,
        ):
            raise ValueError("k/v shape does not match the multi-GPU plan")
        if q_cpu.dtype != self.plan.dtype:
            raise ValueError("input dtype does not match the multi-GPU plan")

    def _run_workers(self, worker, stats: MultiGpuAttentionStats) -> None:
        per_device_stats = stats.per_device
        for schedule in self.plan.schedules:
            per_device_stats.setdefault(str(schedule.device), StreamingAttentionStats())
        start_barrier = threading.Barrier(len(self.runners))

        def run_device(index: int):
            start_barrier.wait()
            device = str(self.plan.schedules[index].device)
            return worker(index, per_device_stats[device])

        started = time.perf_counter()
        futures = [self._executor.submit(run_device, index) for index in range(len(self.runners))]
        for future in futures:
            future.result()
        stats.wall_seconds += time.perf_counter() - started
        stats.per_device = per_device_stats

    def _run_dynamic_workers(
        self,
        worker,
        stats: MultiGpuAttentionStats,
        *,
        consumer_mode: str,
        worker_context=None,
    ) -> None:
        assert self.dynamic_config is not None
        per_device_stats = stats.per_device
        dynamic_stats = stats.dynamic_per_device
        for index, schedule in enumerate(self.plan.schedules):
            device = str(schedule.device)
            per_device_stats.setdefault(device, StreamingAttentionStats())
            dynamic_stats[device] = DynamicDeviceStats()
            self.controllers[index].reset_for_signature(
                self._workload_signature(schedule, consumer_mode)
            )

        cursor = DynamicQueryCursor(
            self.plan.q_bounds,
            self.plan.k_bounds,
            active_workers=len(self.runners),
            tail_balance_factor=self.dynamic_config.tail_balance_factor,
        )
        start_barrier = threading.Barrier(len(self.runners))
        failure_lock = threading.Lock()
        trace_lock = threading.Lock()
        failures: list[Exception] = []

        def run_device(index: int) -> None:
            schedule = self.plan.schedules[index]
            device = str(schedule.device)
            controller = self.controllers[index]
            device_stats = per_device_stats[device]
            summary = dynamic_stats[device]
            start_barrier.wait()
            try:
                context = nullcontext() if worker_context is None else worker_context(index)
                with context:
                    while True:
                        requested_q = controller.q_current
                        task = cursor.claim(index, requested_q)
                        if task is None:
                            break
                        measurement = QueryTaskMeasurement()
                        h2d_before = device_stats.h2d_bytes
                        d2h_before = device_stats.d2h_bytes
                        task_started = time.perf_counter()
                        worker(index, task, device_stats, measurement)
                        host_elapsed = time.perf_counter() - task_started
                        if measurement.elapsed_seconds <= 0:
                            measurement.elapsed_seconds = host_elapsed
                        if measurement.attention_seconds <= 0:
                            measurement.attention_seconds = host_elapsed
                        if measurement.h2d_bytes <= 0:
                            measurement.h2d_bytes = device_stats.h2d_bytes - h2d_before
                        if measurement.d2h_bytes <= 0:
                            measurement.d2h_bytes = device_stats.d2h_bytes - d2h_before
                        if measurement.attention_flops <= 0:
                            measurement.attention_flops = (
                                4
                                * task.q_tokens
                                * task.k_tokens
                                * self.plan.q_heads
                                * self.plan.head_dim
                            )

                        q_before, q_after = controller.observe(
                            measurement,
                            update_compute=not (task.segment_clamped or task.tail_clamped),
                        )
                        summary.task_count += 1
                        summary.q_tokens += task.q_tokens
                        summary.q_tokens_min = (
                            task.q_tokens
                            if summary.q_tokens_min == 0
                            else min(summary.q_tokens_min, task.q_tokens)
                        )
                        summary.q_tokens_max = max(summary.q_tokens_max, task.q_tokens)
                        summary.busy_seconds += measurement.elapsed_seconds
                        summary.attention_seconds += measurement.attention_seconds
                        summary.h2d_seconds += measurement.h2d_seconds
                        summary.d2h_seconds += measurement.d2h_seconds
                        summary.attention_flops += measurement.attention_flops
                        summary.h2d_bytes += measurement.h2d_bytes
                        summary.d2h_bytes += measurement.d2h_bytes
                        if self.dynamic_config.enable_task_trace:
                            trace = DynamicTaskTrace(
                                device=device,
                                segment_id=task.segment_id,
                                q_start=task.q_start,
                                q_stop=task.q_stop,
                                claim_order=task.claim_order,
                                requested_q=requested_q,
                                actual_q=task.q_tokens,
                                h2d_seconds=measurement.h2d_seconds,
                                attention_seconds=measurement.attention_seconds,
                                consumer_seconds=measurement.consumer_seconds,
                                d2h_seconds=measurement.d2h_seconds,
                                elapsed_seconds=measurement.elapsed_seconds,
                                q_before=q_before,
                                q_after=q_after,
                                segment_clamped=task.segment_clamped,
                                tail_clamped=task.tail_clamped,
                            )
                            with trace_lock:
                                stats.task_trace.append(trace)
            except Exception as error:  # noqa: BLE001 - worker boundary propagates the original.
                cursor.cancel()
                with failure_lock:
                    if not failures:
                        failures.append(error)
            finally:
                cursor.retire(index)

        started = time.perf_counter()
        futures = [self._executor.submit(run_device, index) for index in range(len(self.runners))]
        for future in futures:
            future.result()
        stats.wall_seconds += time.perf_counter() - started
        if failures:
            raise failures[0]

        total_q_tokens = sum(item.q_tokens for item in dynamic_stats.values())
        for index, schedule in enumerate(self.plan.schedules):
            summary = dynamic_stats[str(schedule.device)]
            summary.q_tokens_average = (
                summary.q_tokens / summary.task_count if summary.task_count else 0.0
            )
            summary.effective_tflops = (
                summary.attention_flops / summary.attention_seconds / 1e12
                if summary.attention_seconds > 0
                else 0.0
            )
            summary.h2d_gbps = (
                summary.h2d_bytes / summary.h2d_seconds / 1e9 if summary.h2d_seconds > 0 else 0.0
            )
            summary.d2h_gbps = (
                summary.d2h_bytes / summary.d2h_seconds / 1e9 if summary.d2h_seconds > 0 else 0.0
            )
            snapshot = self.controllers[index].snapshot()
            summary.effective_tflops_ema = snapshot.effective_tflops_ema
            summary.h2d_gbps_ema = snapshot.h2d_gbps_ema
            summary.d2h_gbps_ema = snapshot.d2h_gbps_ema
            summary.task_elapsed_ema = snapshot.task_elapsed_ema
            summary.q_current = snapshot.q_current
            summary.work_fraction = summary.q_tokens / total_q_tokens if total_q_tokens else 0.0
        stats.task_trace.sort(key=lambda item: item.claim_order)
        stats.per_device = per_device_stats
        stats.dynamic_per_device = dynamic_stats

    @torch.inference_mode()
    def __call__(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        stats: MultiGpuAttentionStats | None = None,
    ) -> torch.Tensor:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("MultiGpuStreamingAttentionRunner is single-flight")
        try:
            if self.plan.output_mode != "host":
                raise ValueError("a device_consumer plan requires run_with_device_consumers()")
            self._validate_runtime_inputs(q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k)
            pin_output = any(schedule.attention_plan.pin_output for schedule in self.plan.schedules)
            if out is None:
                out = torch.empty(
                    q_cpu.shape,
                    dtype=q_cpu.dtype,
                    device="cpu",
                    pin_memory=pin_output,
                )
            if out.device.type != "cpu" or out.shape != q_cpu.shape or out.dtype != q_cpu.dtype:
                raise ValueError("out must be a CPU tensor matching q shape and dtype")

            stats = MultiGpuAttentionStats() if stats is None else stats

            if self.schedule_mode == "static":

                def static_worker(
                    index: int, device_stats: StreamingAttentionStats
                ) -> torch.Tensor:
                    schedule = self.plan.schedules[index]
                    return self.runners[index].run_query_tasks(
                        q_cpu,
                        k_cpu,
                        v_cpu,
                        schedule.query_tasks,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        out=out,
                        stats=device_stats,
                    )

                self._run_workers(static_worker, stats)
            else:

                def dynamic_worker(
                    index: int,
                    task: QueryTask,
                    device_stats: StreamingAttentionStats,
                    measurement: QueryTaskMeasurement,
                ) -> torch.Tensor:
                    return self.runners[index].run_query_tasks(
                        q_cpu,
                        k_cpu,
                        v_cpu,
                        (task,),
                        softmax_scale=softmax_scale,
                        causal=causal,
                        out=out,
                        stats=device_stats,
                        task_measurement=measurement,
                    )

                self._run_dynamic_workers(
                    dynamic_worker,
                    stats,
                    consumer_mode="host",
                )
            return out
        finally:
            self._run_lock.release()

    @torch.inference_mode()
    def run_with_device_consumers(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        output_consumers: dict[
            torch.device | str,
            DeviceOutputConsumer | TaskDeviceOutputConsumer,
        ],
        device_contexts: dict[torch.device | str, Callable[[], AbstractContextManager]]
        | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: MultiGpuAttentionStats | None = None,
    ) -> None:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("MultiGpuStreamingAttentionRunner is single-flight")
        try:
            if self.plan.output_mode != "device_consumer":
                raise ValueError("run_with_device_consumers requires device_consumer plans")
            self._validate_runtime_inputs(q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k)
            consumers = {
                str(torch.device(device)): consumer for device, consumer in output_consumers.items()
            }
            expected = {str(device) for device in self.plan.devices}
            if set(consumers) != expected:
                raise ValueError(f"output_consumers must contain exactly {sorted(expected)}")
            contexts = {
                str(torch.device(device)): context
                for device, context in ({} if device_contexts is None else device_contexts).items()
            }
            if not set(contexts) <= expected:
                raise ValueError("device_contexts contains a device outside the multi-GPU plan")
            stats = MultiGpuAttentionStats() if stats is None else stats

            if self.schedule_mode == "static":

                def static_worker(index: int, device_stats: StreamingAttentionStats) -> None:
                    schedule = self.plan.schedules[index]
                    context_factory = contexts.get(str(schedule.device))
                    context = nullcontext() if context_factory is None else context_factory()
                    with context:
                        self.runners[index].run_query_tasks_with_device_consumer(
                            q_cpu,
                            k_cpu,
                            v_cpu,
                            schedule.query_tasks,
                            output_consumer=consumers[str(schedule.device)],
                            softmax_scale=softmax_scale,
                            causal=causal,
                            stats=device_stats,
                        )

                self._run_workers(static_worker, stats)
            else:

                def dynamic_worker(
                    index: int,
                    task: QueryTask,
                    device_stats: StreamingAttentionStats,
                    measurement: QueryTaskMeasurement,
                ) -> None:
                    schedule = self.plan.schedules[index]
                    self.runners[index]._run_query_task_with_task_consumer(
                        q_cpu,
                        k_cpu,
                        v_cpu,
                        task,
                        output_consumer=consumers[str(schedule.device)],
                        softmax_scale=softmax_scale,
                        causal=causal,
                        stats=device_stats,
                        task_measurement=measurement,
                    )

                def dynamic_context(index: int):
                    schedule = self.plan.schedules[index]
                    context_factory = contexts.get(str(schedule.device))
                    return nullcontext() if context_factory is None else context_factory()

                self._run_dynamic_workers(
                    dynamic_worker,
                    stats,
                    consumer_mode="device_consumer",
                    worker_context=dynamic_context,
                )
        finally:
            self._run_lock.release()


__all__ = [
    "MultiGpuStreamingAttentionRunner",
]
