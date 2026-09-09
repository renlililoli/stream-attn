"""Deterministic list scheduling and physical buffer lifetimes; no device execution."""

from __future__ import annotations

import heapq
from bisect import bisect_right, insort
from dataclasses import replace
from itertools import islice, pairwise

from .specs import BufferLifetime, ExecutionSpec, ExecutionTrace, OperationEvent


def merge_intervals(intervals) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _first_gap(calendars, resources, earliest, duration):
    start = earliest
    while True:
        blocked_until = start
        for resource in resources:
            calendar = calendars.get(resource, ())
            # Exclusive, non-overlapping intervals are ordered by both start
            # and end. Skip old reservations without scanning the full trace.
            index = max(0, bisect_right(calendar, (start, float("inf"))) - 1)
            for left, right in islice(calendar, index, None):
                if right <= start:
                    continue
                if left >= start + duration:
                    break
                blocked_until = max(blocked_until, right)
                break
        if blocked_until == start:
            return start
        start = blocked_until


def _lifetimes(spec: ExecutionSpec, events: tuple[OperationEvent, ...]):
    by_name = {event.name: event for event in events}
    pools = {pool.name: pool for pool in spec.pools}
    duration = max(event.end_seconds for event in events)
    buffers = []
    for buffer in spec.buffers:
        intervals = merge_intervals(
            (by_name[name].start_seconds, by_name[name].end_seconds) for name in buffer.uses
        )
        start = (
            0.0
            if buffer.persistent
            else (
                by_name[buffer.allocate_before].start_seconds
                if buffer.allocate_before
                else intervals[0][0]
            )
        )
        end = (
            duration
            if buffer.persistent
            else (
                by_name[buffer.release_after].end_seconds
                if buffer.release_after
                else intervals[-1][1]
            )
        )
        if end < start or any(a < start or b > end for a, b in intervals):
            raise ValueError(f"buffer lifetime does not cover every use: {buffer.name}")
        buffers.append(
            BufferLifetime(
                name=buffer.name,
                pool=buffer.pool,
                size_bytes=buffer.size_bytes,
                allocated_bytes=pools[buffer.pool].allocated_bytes(buffer.size_bytes),
                component=buffer.component,
                owner=buffer.owner,
                persistent=buffer.persistent,
                start_seconds=start,
                end_seconds=end,
                active_intervals=intervals,
            )
        )
    return tuple(buffers)


def schedule_execution(spec: ExecutionSpec) -> ExecutionTrace:
    """Schedule operations under the explicitly declared arbitration policy.

    Legacy input_order reserves resource gaps in declaration order. earliest_ready
    dispatches the earliest feasible operation, using input order only for ties;
    future copies cannot reserve compute ahead of a runnable GEMM. Neither policy
    claims global optimality or models concurrent kernels on an exclusive resource.
    Dependencies encode data hazards, stream order, and ring-slot reuse.
    """
    by_name = {op.name: op for op in spec.operations}
    index = {op.name: i for i, op in enumerate(spec.operations)}
    pending = {op.name: len(op.dependencies) for op in spec.operations}
    children: dict[str, list[str]] = {op.name: [] for op in spec.operations}
    for op in spec.operations:
        for dependency in op.dependencies:
            children[dependency].append(op.name)
    chronological = spec.scheduling_policy == "earliest_ready"
    ready = [
        (0.0 if chronological else index[name], index[name])
        for name, count in pending.items()
        if count == 0
    ]
    heapq.heapify(ready)
    calendars: dict[str, list[tuple[float, float]]] = {}
    events: dict[str, OperationEvent] = {}
    while ready:
        _, op_index = heapq.heappop(ready)
        op = spec.operations[op_index]
        earliest = max((events[name].end_seconds for name in op.dependencies), default=0.0)
        start = _first_gap(calendars, op.resources, earliest, op.duration_seconds)
        if chronological and ready and (start, op_index) > ready[0]:
            # Reservations only increase feasible start times. Refresh this lazy
            # lower bound and let an earlier runnable operation go first.
            heapq.heappush(ready, (start, op_index))
            continue
        end = start + op.duration_seconds
        event = OperationEvent(
            name=op.name,
            start_seconds=start,
            end_seconds=end,
            resources=op.resources,
            kind=op.kind,
            component=op.component,
            dependencies=op.dependencies,
            provenance=op.provenance,
            details=op.details,
        )
        events[op.name] = event
        for resource in op.resources:
            insort(calendars.setdefault(resource, []), (start, end))
        for child in children[op.name]:
            pending[child] -= 1
            if pending[child] == 0:
                child_index = index[child]
                earliest_child = max(events[d].end_seconds for d in by_name[child].dependencies)
                heapq.heappush(
                    ready, (earliest_child if chronological else child_index, child_index)
                )
    if len(events) != len(by_name):
        raise ValueError("operation dependencies contain a cycle")
    ordered = tuple(events[op.name] for op in spec.operations)
    return ExecutionTrace(
        name=spec.name,
        pools=spec.pools,
        operations=ordered,
        buffers=_lifetimes(spec, ordered),
        duration_seconds=max(event.end_seconds for event in ordered),
        metadata={**spec.metadata, "scheduling_policy": spec.scheduling_policy},
        assumptions=spec.assumptions,
    )


def trace_from_measurements(
    spec: ExecutionSpec,
    timings: dict[str, tuple[float, float]],
    *,
    provenance: str,
) -> ExecutionTrace:
    """Bind synchronized, same-clock operation timings to declared buffer ownership.

    Only operation times become measured. Buffer sizes/lifetimes still follow
    the supplied spec; this is not an allocator or whole-process memory trace.
    """
    from .specs import finite_number

    predicted = schedule_execution(spec)  # Validate dependencies, including cycles.
    if set(timings) != {op.name for op in spec.operations}:
        raise ValueError("timings must contain exactly the execution's operation names")
    for start, end in timings.values():
        finite_number("start_seconds", start, allow_zero=True)
        finite_number("end_seconds", end, allow_zero=True)
        if start > end:
            raise ValueError("measured operations cannot have negative durations")
    ordered = tuple(
        replace(
            event,
            start_seconds=timings[event.name][0],
            end_seconds=timings[event.name][1],
            provenance=provenance,
        )
        for event in predicted.operations
    )
    for event in ordered:
        if event.kind != "control" and event.start_seconds == event.end_seconds:
            raise ValueError("measured operations must have positive durations")
        if event.kind == "control" and event.start_seconds != event.end_seconds:
            raise ValueError("control milestones must have zero duration")
        if any(timings[name][1] > event.start_seconds for name in event.dependencies):
            raise ValueError(f"measured timing violates a dependency in {event.name}")
    calendars: dict[str, list[tuple[float, float]]] = {}
    for event in ordered:
        for resource in event.resources:
            calendars.setdefault(resource, []).append((event.start_seconds, event.end_seconds))
    for resource, intervals in calendars.items():
        intervals.sort()
        if any(left[1] > right[0] for left, right in pairwise(intervals)):
            raise ValueError(f"measured timing overlaps exclusive resource {resource}")
    return replace(
        predicted,
        operations=ordered,
        buffers=_lifetimes(spec, ordered),
        duration_seconds=max(event.end_seconds for event in ordered),
        source="measured",
        assumptions=spec.assumptions
        + ("Operation timestamps are measured; allocation sizes and lifetimes are declared.",),
    )
