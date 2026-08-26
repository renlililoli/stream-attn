from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise

import torch

from ..config import StreamingAttentionConfig
from ..planner import AttentionPlan, build_plan
from ..stats import MultiGpuAttentionStats, StreamingAttentionStats
from ..validation import validate_host_qkv
from .runner import StreamingAttentionRunner
from .tasks import QueryTask, build_query_tasks


@dataclass(frozen=True)
class MultiGpuDeviceSpec:
    """One device's independent workspace and measured performance inputs."""

    device: torch.device | str
    config: StreamingAttentionConfig
    compute_tflops: float
    h2d_gbps: float
    d2h_gbps: float | None = None
    fixed_q_block_overhead_us: float = 0.0

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


@dataclass(frozen=True)
class DeviceQuerySchedule:
    device: torch.device
    attention_plan: AttentionPlan
    config: StreamingAttentionConfig
    q_range_start: int
    q_range_stop: int
    query_tasks: tuple[QueryTask, ...]
    estimated_seconds: float

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
        q_chunk_tokens=plan.q_chunk_tokens,
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
) -> MultiGpuAttentionPlan:
    """Build one immutable heterogeneous Q-sharding schedule before execution."""

    if len(devices) < 2:
        raise ValueError("multi-GPU planning requires at least two devices")
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
        normalized = MultiGpuDeviceSpec(
            device=device,
            config=spec.config,
            compute_tflops=spec.compute_tflops,
            h2d_gbps=spec.h2d_gbps,
            d2h_gbps=spec.d2h_gbps,
            fixed_q_block_overhead_us=spec.fixed_q_block_overhead_us,
        )
        normalized_specs.append(normalized)
        plans.append(
            build_plan(
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                max_q_tokens=max_q_tokens,
                max_kv_tokens=max_kv_tokens,
                config=spec.config,
            )
        )
    output_modes = {plan.output_mode for plan in plans}
    if len(output_modes) != 1:
        raise ValueError("all multi-GPU device plans must use the same output_mode")

    ranges = _partition_query_ranges(
        q_bounds=q_bounds,
        k_bounds=k_bounds,
        plans=plans,
        specs=normalized_specs,
    )
    schedules = []
    for spec, plan, (range_start, range_stop) in zip(normalized_specs, plans, ranges):
        tasks = build_query_tasks(
            q_bounds,
            k_bounds,
            q_chunk_tokens=plan.q_chunk_tokens,
            range_start=range_start,
            range_stop=range_stop,
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
    )


class MultiGpuStreamingAttentionRunner:
    """Single-flight executor for an immutable static multi-GPU Q schedule."""

    def __init__(
        self,
        plan: MultiGpuAttentionPlan,
        *,
        runner_overrides: dict[torch.device | str, StreamingAttentionRunner] | None = None,
    ) -> None:
        self.plan = plan
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
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.runners),
            thread_name_prefix="seqattn-gpu",
        )
        self._run_lock = threading.Lock()

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
            raise ValueError("runtime cu_seqlens do not match the static multi-GPU plan")
        if q_cpu.shape != (
            self.plan.max_q_tokens,
            self.plan.q_heads,
            self.plan.head_dim,
        ):
            raise ValueError("q shape does not match the static multi-GPU plan")
        if k_cpu.shape != (
            self.plan.max_kv_tokens,
            self.plan.kv_heads,
            self.plan.head_dim,
        ):
            raise ValueError("k/v shape does not match the static multi-GPU plan")
        if q_cpu.dtype != self.plan.dtype:
            raise ValueError("input dtype does not match the static multi-GPU plan")

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

            def worker(index: int, device_stats: StreamingAttentionStats) -> torch.Tensor:
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

            self._run_workers(worker, stats)
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
        output_consumers: dict[torch.device | str, object],
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

            def worker(index: int, device_stats: StreamingAttentionStats) -> None:
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

            self._run_workers(worker, stats)
        finally:
            self._run_lock.release()


__all__ = [
    "DeviceQuerySchedule",
    "MultiGpuAttentionPlan",
    "MultiGpuDeviceSpec",
    "MultiGpuStreamingAttentionRunner",
    "build_multi_gpu_plan",
]
