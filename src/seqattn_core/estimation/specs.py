"""Device-neutral, physical-allocation inputs for offline estimation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Literal


def positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")


def finite_number(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )


def names(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must be a tuple of non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class MemoryPool:
    """One physical pool; shared/unified address spaces must use the same name."""

    name: str
    capacity_bytes: int | None = None
    allocation_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        names("pool name", (self.name,))
        positive_int("allocation_alignment_bytes", self.allocation_alignment_bytes)
        if self.capacity_bytes is not None:
            positive_int("capacity_bytes", self.capacity_bytes)

    def allocated_bytes(self, size_bytes: int) -> int:
        alignment = self.allocation_alignment_bytes
        return (size_bytes + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class RateProfile:
    """Effective FLOP/s or byte/s for a specific implementation and tile shape.

    Operations reserve every named resource exclusively. Sharing a resource
    models contention; separate resources explicitly permit overlap.
    """

    name: str
    work_per_second: float
    resources: tuple[str, ...]
    kind: Literal["compute", "io"] = "compute"
    latency_seconds: float = 0.0
    provenance: str = "user-supplied estimate"

    def __post_init__(self) -> None:
        names("profile name", (self.name,))
        names("resources", self.resources)
        if not self.resources:
            raise ValueError("a rate profile requires at least one resource")
        if self.kind not in {"compute", "io"}:
            raise ValueError("kind must be compute or io")
        finite_number("work_per_second", self.work_per_second)
        finite_number("latency_seconds", self.latency_seconds, allow_zero=True)

    def seconds(self, work: float) -> float:
        finite_number("work", work)
        result = self.latency_seconds + work / self.work_per_second
        finite_number("estimated duration", result)
        return result


@dataclass(frozen=True)
class OperationSpec:
    name: str
    duration_seconds: float
    resources: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    kind: Literal["compute", "io", "control"] = "compute"
    component: str = "other"
    provenance: str = "user-supplied estimate"
    details: str = ""

    def __post_init__(self) -> None:
        names("operation name", (self.name,))
        names("resources", self.resources)
        names("dependencies", self.dependencies)
        if self.kind == "control":
            if self.duration_seconds != 0 or self.resources:
                raise ValueError("control milestones must have zero duration and no resources")
        else:
            if not self.resources or self.kind not in {"compute", "io"}:
                raise ValueError("operations require resources and kind compute or io")
            finite_number("duration_seconds", self.duration_seconds)
        names("component", (self.component,))


@dataclass(frozen=True)
class BufferSpec:
    """A physical allocation, possibly shared by multiple logical views.

    ``uses`` lists all operations touching the allocation, including async
    copies. Transients remain allocated between their first and last use;
    persistent allocations span the complete trace. Do not add another buffer
    for an alias of this allocation. Size includes layout/padding, before the
    pool's allocation alignment. Ownership is informational; all entries count.
    """

    name: str
    pool: str
    size_bytes: int
    component: str
    uses: tuple[str, ...] = ()
    persistent: bool = False
    owner: str = "operator"
    allocate_before: str | None = None
    release_after: str | None = None

    def __post_init__(self) -> None:
        for label in ("name", "pool", "component", "owner"):
            names(label, (getattr(self, label),))
        names("uses", self.uses)
        positive_int("size_bytes", self.size_bytes)
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a bool")
        if (
            not self.uses
            and not self.persistent
            and not (self.allocate_before and self.release_after)
        ):
            raise ValueError("a transient buffer requires uses or explicit lifetime anchors")
        if self.persistent and (self.allocate_before or self.release_after):
            raise ValueError("persistent buffers cannot have transient lifetime anchors")


@dataclass(frozen=True)
class ExecutionSpec:
    name: str
    pools: tuple[MemoryPool, ...]
    operations: tuple[OperationSpec, ...]
    buffers: tuple[BufferSpec, ...]
    metadata: dict[str, object] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names("execution name", (self.name,))
        if not self.pools or not self.operations:
            raise ValueError("an execution requires memory pools and operations")
        for label, entries in (
            ("pools", self.pools),
            ("operations", self.operations),
            ("buffers", self.buffers),
        ):
            names(label, tuple(entry.name for entry in entries))
        pool_names = {pool.name for pool in self.pools}
        operation_names = {op.name for op in self.operations}
        for op in self.operations:
            if set(op.dependencies) - operation_names:
                raise ValueError(f"unknown dependency in {op.name}")
        for buffer in self.buffers:
            references = (*buffer.uses, buffer.allocate_before, buffer.release_after)
            if (
                buffer.pool not in pool_names
                or {name for name in references if name} - operation_names
            ):
                raise ValueError(f"unknown pool or operation in buffer {buffer.name}")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")
        json.dumps(self.metadata, allow_nan=False)


@dataclass(frozen=True)
class OperationEvent:
    name: str
    start_seconds: float
    end_seconds: float
    resources: tuple[str, ...]
    kind: str
    component: str
    dependencies: tuple[str, ...]
    provenance: str
    details: str


@dataclass(frozen=True)
class BufferLifetime:
    name: str
    pool: str
    size_bytes: int
    allocated_bytes: int
    component: str
    owner: str
    persistent: bool
    start_seconds: float
    end_seconds: float
    active_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ExecutionTrace:
    name: str
    pools: tuple[MemoryPool, ...]
    operations: tuple[OperationEvent, ...]
    buffers: tuple[BufferLifetime, ...]
    duration_seconds: float
    metadata: dict[str, object]
    assumptions: tuple[str, ...]
    source: Literal["predicted", "measured"] = "predicted"
