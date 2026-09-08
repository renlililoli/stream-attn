"""Memory/latency selection across explicitly supplied execution candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .memory import TimelineStats, memory_statistics
from .schedule import schedule_execution
from .specs import ExecutionSpec, ExecutionTrace, finite_number


@dataclass(frozen=True)
class CandidateEstimate:
    trace: ExecutionTrace
    stats: TimelineStats
    objective_pool: str
    objective_peak_bytes: int | None = None
    objective_owners: frozenset[str] | None = None

    @property
    def peak_bytes(self) -> int:
        if self.objective_peak_bytes is not None:
            return self.objective_peak_bytes
        return next(
            pool.peak_bytes for pool in self.stats.pools if pool.pool == self.objective_pool
        )


@dataclass(frozen=True)
class ActivationMemoryEstimate:
    candidates: tuple[CandidateEstimate, ...]
    pareto: tuple[CandidateEstimate, ...]
    minimum_capacity: CandidateEstimate | None
    selected: CandidateEstimate | None
    latency_limit_seconds: float
    reference_latency_seconds: float
    target_throughput_fraction: float


def estimate_activation_memory(
    candidates: Iterable[ExecutionSpec],
    *,
    objective_pool: str,
    target_throughput_fraction: float = 0.95,
    max_latency_seconds: float | None = None,
    objective_owners: frozenset[str] | None = None,
) -> ActivationMemoryEstimate:
    """Minimize one pool's peak subject to all pool capacities and whole-trace latency.

    The throughput reference is the fastest supplied candidate *before* capacity
    filtering; an insufficient capacity must not silently lower the target.
    Candidates must represent the same workload. Minimality is only over the
    supplied candidates and their specified schedules, not all possible tiles.
    """
    finite_number("target_throughput_fraction", target_throughput_fraction)
    if target_throughput_fraction > 1:
        raise ValueError("target_throughput_fraction must be within (0, 1]")
    if max_latency_seconds is not None:
        finite_number("max_latency_seconds", max_latency_seconds)
    estimates = []
    for spec in candidates:
        if objective_pool not in {pool.name for pool in spec.pools}:
            raise ValueError(f"unknown objective_pool {objective_pool!r} in {spec.name}")
        trace = schedule_execution(spec)
        objective_peak = None
        if objective_owners is not None:
            scoped = memory_statistics(trace, owners=objective_owners)
            objective_peak = next(
                pool.peak_bytes for pool in scoped.pools if pool.pool == objective_pool
            )
        estimates.append(
            CandidateEstimate(
                trace, memory_statistics(trace), objective_pool, objective_peak, objective_owners
            )
        )
    if not estimates:
        raise ValueError("at least one candidate is required")
    reference = min(item.trace.duration_seconds for item in estimates)
    limit = reference / target_throughput_fraction
    if max_latency_seconds is not None:
        limit = min(limit, max_latency_seconds)
    feasible = [item for item in estimates if item.stats.fits_capacity]
    ordered = sorted(feasible, key=lambda item: (item.peak_bytes, item.trace.duration_seconds))
    pareto = []
    fastest = float("inf")
    for item in ordered:
        if item.trace.duration_seconds < fastest:
            pareto.append(item)
            fastest = item.trace.duration_seconds
    selected = next((item for item in ordered if item.trace.duration_seconds <= limit), None)
    return ActivationMemoryEstimate(
        candidates=tuple(estimates),
        pareto=tuple(pareto),
        minimum_capacity=ordered[0] if ordered else None,
        selected=selected,
        latency_limit_seconds=limit,
        reference_latency_seconds=reference,
        target_throughput_fraction=target_throughput_fraction,
    )
