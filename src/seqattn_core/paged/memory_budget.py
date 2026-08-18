from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class HostMemorySnapshot:
    operator_host_allocated_bytes: int
    operator_host_peak_bytes: int
    pinned_allocated_bytes: int
    pinned_peak_bytes: int
    direct_io_bounce_allocated_bytes: int
    direct_io_bounce_peak_bytes: int
    dram_cache_allocated_bytes: int
    dram_cache_peak_bytes: int
    metadata_allocated_bytes: int
    host_memory_budget_bytes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class HostMemoryPlan:
    """Tracks and enforces every host allocation owned by a paged runner."""

    _CATEGORIES: ClassVar[frozenset[str]] = frozenset({"pinned", "bounce", "cache", "metadata"})

    def __init__(
        self,
        *,
        total_budget_bytes: int,
        pinned_limit_bytes: int,
        bounce_limit_bytes: int,
        metadata_margin_bytes: int,
    ) -> None:
        if total_budget_bytes <= 0:
            raise ValueError("total_budget_bytes must be positive")
        if min(pinned_limit_bytes, bounce_limit_bytes, metadata_margin_bytes) < 0:
            raise ValueError("host memory limits must be non-negative")
        if pinned_limit_bytes + bounce_limit_bytes + metadata_margin_bytes >= total_budget_bytes:
            raise ValueError("host reservations leave no room for the DRAM cache")
        self.total_budget_bytes = total_budget_bytes
        self.pinned_limit_bytes = pinned_limit_bytes
        self.bounce_limit_bytes = bounce_limit_bytes
        self.metadata_margin_bytes = metadata_margin_bytes
        self.cache_limit_bytes = (
            total_budget_bytes - pinned_limit_bytes - bounce_limit_bytes - metadata_margin_bytes
        )
        self._current = {name: 0 for name in self._CATEGORIES}
        self._peak = {name: 0 for name in self._CATEGORIES}
        self._peak_total = 0
        self._lock = threading.Lock()
        self.register("metadata", metadata_margin_bytes)

    def _category_limit(self, category: str) -> int:
        if category == "pinned":
            return self.pinned_limit_bytes
        if category == "bounce":
            return self.bounce_limit_bytes
        if category == "cache":
            return self.cache_limit_bytes
        if category == "metadata":
            return self.metadata_margin_bytes
        raise ValueError(f"unknown host allocation category: {category}")

    def register(self, category: str, size_bytes: int) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"unknown host allocation category: {category}")
        if size_bytes < 0:
            raise ValueError("allocation size must be non-negative")
        with self._lock:
            category_total = self._current[category] + size_bytes
            limit = self._category_limit(category)
            if category_total > limit:
                raise MemoryError(
                    f"{category} allocation exceeds its limit: {category_total} > {limit} bytes"
                )
            total = sum(self._current.values()) + size_bytes
            if total > self.total_budget_bytes:
                raise MemoryError(
                    "operator-owned host allocation exceeds the configured budget: "
                    f"{total} > {self.total_budget_bytes} bytes"
                )
            self._current[category] = category_total
            self._peak[category] = max(self._peak[category], category_total)
            self._peak_total = max(self._peak_total, total)

    def release(self, category: str, size_bytes: int) -> None:
        if category not in self._CATEGORIES:
            raise ValueError(f"unknown host allocation category: {category}")
        if size_bytes < 0:
            raise ValueError("allocation size must be non-negative")
        with self._lock:
            if size_bytes > self._current[category]:
                raise RuntimeError(f"released more {category} memory than was registered")
            self._current[category] -= size_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return sum(self._current.values())

    def snapshot(self) -> HostMemorySnapshot:
        with self._lock:
            return HostMemorySnapshot(
                operator_host_allocated_bytes=sum(self._current.values()),
                operator_host_peak_bytes=self._peak_total,
                pinned_allocated_bytes=self._current["pinned"],
                pinned_peak_bytes=self._peak["pinned"],
                direct_io_bounce_allocated_bytes=self._current["bounce"],
                direct_io_bounce_peak_bytes=self._peak["bounce"],
                dram_cache_allocated_bytes=self._current["cache"],
                dram_cache_peak_bytes=self._peak["cache"],
                metadata_allocated_bytes=self._current["metadata"],
                host_memory_budget_bytes=self.total_budget_bytes,
            )
