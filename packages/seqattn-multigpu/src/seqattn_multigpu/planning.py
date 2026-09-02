from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal

import torch
from seqattn_core._plugin_api import (
    AttentionPlan,
    QueryTask,
    StreamingAttentionConfig,
    build_attention_plan,
    build_query_tasks,
)

from .dynamic import DynamicScheduleConfig


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
        initial_plan = build_attention_plan(
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
            else build_attention_plan(
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


__all__ = [
    "DeviceQuerySchedule",
    "MultiGpuAttentionPlan",
    "MultiGpuDeviceSpec",
    "build_multi_gpu_plan",
]
