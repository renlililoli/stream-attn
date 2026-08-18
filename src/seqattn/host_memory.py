"""Compatibility facade for paged host-memory accounting."""

from seqattn_core.paged.memory_budget import HostMemoryPlan, HostMemorySnapshot

__all__ = ["HostMemoryPlan", "HostMemorySnapshot"]
