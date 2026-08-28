from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal

import torch
from seqattn_core._plugin_api import (
    AttentionPlan,
    DeviceOutputConsumer,
    QueryTask,
    QueryTaskMeasurement,
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    TaskDeviceOutputConsumer,
    build_plan,
    build_query_tasks,
    validate_host_qkv,
)

from .dynamic import (
    DynamicQController,
    DynamicQueryCursor,
    DynamicScheduleConfig,
    DynamicWorkloadSignature,
)
from .stats import DynamicDeviceStats, DynamicTaskTrace, MultiGpuAttentionStats


@dataclass(frozen=True)
class MultiGpuDeviceSpec:
    """One device's independent workspace and measured performance inputs."""

    device: torch.device | str
    config: StreamingAttentionConfig
    compute_tflops: float
    h2d_gbps: float
    d2h_gbps: float | None = None
    fixed_q_block_overhead_us: float = 0.0
    q_chunk_tokens: int | None = None
    q_min_tokens: int | None = None
    q_capacity_tokens: int | None = None

    def validate(self) -> None:
        device = torch.device(self.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("multi-GPU device specs require explicit cuda:N devices")
        self.config.validate()
        if self.config.backend == "reference":
            raise ValueError("multi-GPU execution requires a CUDA backend")
        if self.compute_tflops <= 0:
            raise ValueError("compute_tflops must be positive")
        if self.h2d_gbps <= 0:
            raise ValueError("h2d_gbps must be positive")
        if self.d2h_gbps is not None and self.d2h_gbps <= 0:
            raise ValueError("d2h_gbps must be positive when provided")
        if self.fixed_q_block_overhead_us < 0:
            raise ValueError("fixed_q_block_overhead_us must be non-negative")
        for name, value in (
            ("q_chunk_tokens", self.q_chunk_tokens),
            ("q_min_tokens", self.q_min_tokens),
            ("q_capacity_tokens", self.q_capacity_tokens),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")


@dataclass(frozen=True)
class DeviceQuerySchedule:
    device: torch.device
    attention_plan: AttentionPlan
    config: StreamingAttentionConfig
    q_range_start: int
    q_range_stop: int
    query_tasks: tuple[QueryTask, ...]
    estimated_seconds: float
    device_spec: MultiGpuDeviceSpec
    initial_q_tokens: int
    q_min_tokens: int
    q_capacity_tokens: int

    @property
    def q_tokens(self) -> int:
        return self.q_range_stop - self.q_range_start


@dataclass(frozen=True)
class MultiGpuAttentionPlan:
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    max_q_tokens: int
    max_kv_tokens: int
    output_mode: str
    q_bounds: tuple[int, ...]
    k_bounds: tuple[int, ...]
    schedules: tuple[DeviceQuerySchedule, ...]
    estimated_makespan_seconds: float
    schedule_mode: Literal["static", "dynamic"] = "static"
    dynamic_config: DynamicScheduleConfig | None = None

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(schedule.device for schedule in self.schedules)


def _bounds_from_cu(name: str, cu_seqlens: torch.Tensor, total_tokens: int) -> list[int]:
    if cu_seqlens.device.type != "cpu" or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional CPU tensor")
    bounds = cu_seqlens.to(dtype=torch.int64).tolist()
    if bounds[0] != 0 or bounds[-1] != total_tokens:
        raise ValueError(f"{name} must span [0, {total_tokens}]")
    if any(stop < start for start, stop in pairwise(bounds)):
        raise ValueError(f"{name} must be non-decreasing")
    return bounds


def _task_seconds(
    task: QueryTask,
    *,
    plan: AttentionPlan,
    spec: MultiGpuDeviceSpec,
) -> float:
    element_size = torch.empty((), dtype=plan.dtype).element_size()
    flops = 4 * task.q_tokens * task.k_tokens * plan.q_heads * plan.head_dim
    compute_seconds = flops / (spec.compute_tflops * 1e12)
    q_bytes = task.q_tokens * plan.q_heads * plan.head_dim * element_size
    kv_bytes = 2 * task.k_tokens * plan.kv_heads * plan.head_dim * element_size
    h2d_seconds = (q_bytes + kv_bytes) / (spec.h2d_gbps * 1e9)
    d2h_gbps = spec.h2d_gbps if spec.d2h_gbps is None else spec.d2h_gbps
    d2h_seconds = q_bytes / (d2h_gbps * 1e9)
    return max(compute_seconds, h2d_seconds, d2h_seconds) + spec.fixed_q_block_overhead_us * 1e-6


def _range_seconds(
    range_start: int,
    range_stop: int,
    *,
    q_bounds: list[int],
    k_bounds: list[int],
    plan: AttentionPlan,
    spec: MultiGpuDeviceSpec,
) -> float:
    tasks = build_query_tasks(
        q_bounds,
        k_bounds,
        q_chunk_tokens=(
            plan.q_chunk_tokens if spec.q_chunk_tokens is None else spec.q_chunk_tokens
        ),
        range_start=range_start,
        range_stop=range_stop,
    )
    return sum(_task_seconds(task, plan=plan, spec=spec) for task in tasks)


def _candidate_cuts(q_bounds: list[int], plans: Sequence[AttentionPlan]) -> list[int]:
    alignment = min(plan.block_m for plan in plans)
    candidates = {q_bounds[0], q_bounds[-1]}
    for start, stop in pairwise(q_bounds):
        candidates.add(start)
        candidates.add(stop)
        candidates.update(range(start + alignment, stop, alignment))
    return sorted(candidates)


def _furthest_cut(
    *,
    start_index: int,
    max_index: int,
    candidates: list[int],
    threshold: float,
    q_bounds: list[int],
    k_bounds: list[int],
    plan: AttentionPlan,
    spec: MultiGpuDeviceSpec,
) -> int | None:
    low = start_index + 1
    high = max_index
    best = None
    while low <= high:
        middle = (low + high) // 2
        seconds = _range_seconds(
            candidates[start_index],
            candidates[middle],
            q_bounds=q_bounds,
            k_bounds=k_bounds,
            plan=plan,
            spec=spec,
        )
        if seconds <= threshold:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _partition_for_threshold(
    threshold: float,
    *,
    candidates: list[int],
    q_bounds: list[int],
    k_bounds: list[int],
    plans: Sequence[AttentionPlan],
    specs: Sequence[MultiGpuDeviceSpec],
) -> list[int] | None:
    cuts = [0]
    current = 0
    last_candidate = len(candidates) - 1
    for index, (plan, spec) in enumerate(zip(plans[:-1], specs[:-1])):
        remaining_devices = len(plans) - index - 1
        cut = _furthest_cut(
            start_index=current,
            max_index=last_candidate - remaining_devices,
            candidates=candidates,
            threshold=threshold,
            q_bounds=q_bounds,
            k_bounds=k_bounds,
            plan=plan,
            spec=spec,
        )
        if cut is None:
            return None
        cuts.append(cut)
        current = cut

    last_seconds = _range_seconds(
        candidates[current],
        candidates[-1],
        q_bounds=q_bounds,
        k_bounds=k_bounds,
        plan=plans[-1],
        spec=specs[-1],
    )
    if current >= last_candidate or last_seconds > threshold:
        return None
    cuts.append(last_candidate)
    return cuts


def _partition_query_ranges(
    *,
    q_bounds: list[int],
    k_bounds: list[int],
    plans: Sequence[AttentionPlan],
    specs: Sequence[MultiGpuDeviceSpec],
) -> list[tuple[int, int]]:
    candidates = _candidate_cuts(q_bounds, plans)
    if len(candidates) <= len(plans):
        raise ValueError("the query sequence is too short to assign work to every GPU")

    lower = 0.0
    upper = max(
        _range_seconds(
            0,
            q_bounds[-1],
            q_bounds=q_bounds,
            k_bounds=k_bounds,
            plan=plan,
            spec=spec,
        )
        for plan, spec in zip(plans, specs)
    )
    best_cuts = None
    for _ in range(60):
        threshold = (lower + upper) / 2
        cuts = _partition_for_threshold(
            threshold,
            candidates=candidates,
            q_bounds=q_bounds,
            k_bounds=k_bounds,
            plans=plans,
            specs=specs,
        )
        if cuts is None:
            lower = threshold
        else:
            upper = threshold
            best_cuts = cuts
    if best_cuts is None:
        raise RuntimeError("failed to construct a static multi-GPU query partition")
    return [(candidates[start], candidates[stop]) for start, stop in pairwise(best_cuts)]


def build_multi_gpu_plan(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    max_q_tokens: int,
    max_kv_tokens: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    devices: Sequence[MultiGpuDeviceSpec],
    schedule_mode: Literal["static", "dynamic"] = "static",
    dynamic_config: DynamicScheduleConfig | None = None,
) -> MultiGpuAttentionPlan:
    """Build immutable per-device workspaces and a static or dynamic Q schedule."""

    if len(devices) < 2:
        raise ValueError("multi-GPU planning requires at least two devices")
    if schedule_mode not in {"static", "dynamic"}:
        raise ValueError("schedule_mode must be 'static' or 'dynamic'")
    if dynamic_config is not None:
        dynamic_config.validate()
    if schedule_mode == "dynamic" and dynamic_config is None:
        dynamic_config = DynamicScheduleConfig()
    q_bounds = _bounds_from_cu("cu_seqlens_q", cu_seqlens_q, max_q_tokens)
    k_bounds = _bounds_from_cu("cu_seqlens_k", cu_seqlens_k, max_kv_tokens)
    if len(q_bounds) != len(k_bounds):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch size")
    for index, ((q_start, q_stop), (k_start, k_stop)) in enumerate(
        zip(pairwise(q_bounds), pairwise(k_bounds))
    ):
        if q_stop > q_start and k_stop == k_start:
            raise ValueError(f"sequence {index} has queries but no keys")

    normalized_specs = []
    seen_devices = set()
    plans = []
    for spec in devices:
        spec.validate()
        device = torch.device(spec.device)
        if device in seen_devices:
            raise ValueError(f"duplicate multi-GPU device: {device}")
        seen_devices.add(device)
        requested_config = (
            spec.config
            if spec.q_chunk_tokens is None
            else replace(spec.config, q_chunk_tokens=spec.q_chunk_tokens)
        )
        initial_plan = build_plan(
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            max_q_tokens=max_q_tokens,
            max_kv_tokens=max_kv_tokens,
            config=requested_config,
        )
        initial_q_tokens = initial_plan.q_chunk_tokens
        q_capacity_tokens = (
            initial_q_tokens if spec.q_capacity_tokens is None else spec.q_capacity_tokens
        )
        q_capacity_tokens = min(q_capacity_tokens, max_q_tokens)
        q_min_tokens = (
            min(initial_plan.block_m, max_q_tokens)
            if spec.q_min_tokens is None
            else spec.q_min_tokens
        )
        if q_min_tokens > q_capacity_tokens:
            raise ValueError("q_min_tokens cannot exceed q_capacity_tokens")
        if initial_q_tokens > q_capacity_tokens:
            raise ValueError("q_chunk_tokens cannot exceed q_capacity_tokens")
        workspace_config = (
            requested_config
            if q_capacity_tokens == initial_plan.q_chunk_tokens
            else replace(requested_config, q_chunk_tokens=q_capacity_tokens)
        )
        plan = (
            initial_plan
            if workspace_config == requested_config
            else build_plan(
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                max_q_tokens=max_q_tokens,
                max_kv_tokens=max_kv_tokens,
                config=workspace_config,
            )
        )
        if q_min_tokens > plan.q_chunk_tokens:
            raise ValueError("q_min_tokens cannot exceed the aligned Q capacity")
        normalized = MultiGpuDeviceSpec(
            device=device,
            config=workspace_config,
            compute_tflops=spec.compute_tflops,
            h2d_gbps=spec.h2d_gbps,
            d2h_gbps=spec.d2h_gbps,
            fixed_q_block_overhead_us=spec.fixed_q_block_overhead_us,
            q_chunk_tokens=initial_q_tokens,
            q_min_tokens=q_min_tokens,
            q_capacity_tokens=plan.q_chunk_tokens,
        )
        normalized_specs.append(normalized)
        plans.append(plan)
    output_modes = {plan.output_mode for plan in plans}
    if len(output_modes) != 1:
        raise ValueError("all multi-GPU device plans must use the same output_mode")

    ranges = (
        _partition_query_ranges(
            q_bounds=q_bounds,
            k_bounds=k_bounds,
            plans=plans,
            specs=normalized_specs,
        )
        if schedule_mode == "static"
        else [(0, max_q_tokens) for _ in plans]
    )
    schedules = []
    for spec, plan, (range_start, range_stop) in zip(normalized_specs, plans, ranges):
        tasks = (
            build_query_tasks(
                q_bounds,
                k_bounds,
                q_chunk_tokens=spec.q_chunk_tokens,
                range_start=range_start,
                range_stop=range_stop,
            )
            if schedule_mode == "static"
            else ()
        )
        estimated_seconds = sum(_task_seconds(task, plan=plan, spec=spec) for task in tasks)
        schedules.append(
            DeviceQuerySchedule(
                device=torch.device(spec.device),
                attention_plan=plan,
                config=spec.config,
                q_range_start=range_start,
                q_range_stop=range_stop,
                query_tasks=tasks,
                estimated_seconds=estimated_seconds,
                device_spec=spec,
                initial_q_tokens=spec.q_chunk_tokens,
                q_min_tokens=spec.q_min_tokens,
                q_capacity_tokens=spec.q_capacity_tokens,
            )
        )

    return MultiGpuAttentionPlan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        max_q_tokens=max_q_tokens,
        max_kv_tokens=max_kv_tokens,
        output_mode=plans[0].output_mode,
        q_bounds=tuple(q_bounds),
        k_bounds=tuple(k_bounds),
        schedules=tuple(schedules),
        estimated_makespan_seconds=max(schedule.estimated_seconds for schedule in schedules),
        schedule_mode=schedule_mode,
        dynamic_config=dynamic_config,
    )


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
                runner = StreamingAttentionRunner(schedule.attention_plan, schedule.config)
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
        self._executor.shutdown(wait=True)

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
            pin_output = any(schedule.config.pin_output for schedule in self.plan.schedules)
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
    "DeviceQuerySchedule",
    "DynamicScheduleConfig",
    "MultiGpuAttentionPlan",
    "MultiGpuDeviceSpec",
    "MultiGpuStreamingAttentionRunner",
    "build_multi_gpu_plan",
]
