"""Event-sweep memory statistics with half-open allocation lifetimes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schedule import merge_intervals
from .specs import ExecutionTrace


@dataclass(frozen=True)
class MemoryPoint:
    seconds: float
    total_bytes: int
    active_bytes: int
    by_component: dict[str, int]


@dataclass(frozen=True)
class PoolMemoryStats:
    pool: str
    capacity_bytes: int | None
    peak_bytes: int
    peak_seconds: float
    components_at_peak: dict[str, int]
    component_peak_bytes: dict[str, int]
    average_bytes: float
    byte_seconds: float
    active_peak_bytes: int
    points: tuple[MemoryPoint, ...]

    @property
    def exceeds_capacity(self) -> bool:
        return self.capacity_bytes is not None and self.peak_bytes > self.capacity_bytes


@dataclass(frozen=True)
class TimelineStats:
    pools: tuple[PoolMemoryStats, ...]
    resource_busy_seconds: dict[str, float]
    compute_seconds: float
    io_seconds: float
    compute_io_overlap_seconds: float

    @property
    def fits_capacity(self) -> bool:
        return not any(pool.exceeds_capacity for pool in self.pools)


def _intersection_seconds(left, right) -> float:
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def memory_statistics(
    trace: ExecutionTrace,
    *,
    owners: frozenset[str] | None = None,
) -> TimelineStats:
    """Report physical occupancy; optionally restrict the explicitly declared owners."""
    pools = []
    for pool in trace.pools:
        buffers = [
            buffer
            for buffer in trace.buffers
            if buffer.pool == pool.name and (owners is None or buffer.owner in owners)
        ]
        components = sorted({buffer.component for buffer in buffers})
        deltas: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        active: dict[float, int] = defaultdict(int)
        for buffer in buffers:
            deltas[buffer.start_seconds][buffer.component] += buffer.allocated_bytes
            # Persistent lifetimes are clipped to the reporting window, not
            # freed when the last operation completes. Keep the terminal value.
            if not buffer.persistent:
                deltas[buffer.end_seconds][buffer.component] -= buffer.allocated_bytes
            for start, end in buffer.active_intervals:
                active[start] += buffer.allocated_bytes
                active[end] -= buffer.allocated_bytes
        # Batch releases and allocations at identical timestamps. There is no
        # synthetic peak when one allocation is replaced by another at that time.
        times = sorted({0.0, trace.duration_seconds, *deltas, *active})
        current = dict.fromkeys(components, 0)
        component_peaks = dict(current)
        peak_components = dict(current)
        total = active_total = peak = active_peak = 0
        peak_time = previous = area = 0.0
        points = []
        for seconds in times:
            area += total * (seconds - previous)
            for component, delta in deltas[seconds].items():
                current[component] += delta
                component_peaks[component] = max(component_peaks[component], current[component])
            total = sum(current.values())
            active_total += active[seconds]
            active_peak = max(active_peak, active_total)
            if total > peak:
                peak, peak_time, peak_components = total, seconds, dict(current)
            points.append(MemoryPoint(seconds, total, active_total, dict(current)))
            previous = seconds
        pools.append(
            PoolMemoryStats(
                pool=pool.name,
                capacity_bytes=pool.capacity_bytes,
                peak_bytes=peak,
                peak_seconds=peak_time,
                components_at_peak=peak_components,
                component_peak_bytes=component_peaks,
                average_bytes=area / trace.duration_seconds,
                byte_seconds=area,
                active_peak_bytes=active_peak,
                points=tuple(points),
            )
        )
    resources = sorted({resource for op in trace.operations for resource in op.resources})
    busy = {
        resource: sum(
            end - start
            for start, end in merge_intervals(
                (op.start_seconds, op.end_seconds)
                for op in trace.operations
                if resource in op.resources
            )
        )
        for resource in resources
    }
    compute, io = (
        merge_intervals(
            (op.start_seconds, op.end_seconds) for op in trace.operations if op.kind == kind
        )
        for kind in ("compute", "io")
    )
    return TimelineStats(
        pools=tuple(pools),
        resource_busy_seconds=busy,
        compute_seconds=sum(end - start for start, end in compute),
        io_seconds=sum(end - start for start, end in io),
        compute_io_overlap_seconds=_intersection_seconds(compute, io),
    )
