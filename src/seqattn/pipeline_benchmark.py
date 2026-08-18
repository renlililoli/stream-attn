"""Compatibility facade for the projected attention benchmark."""

from seqattn_core.benchmarking.common import (
    MemorySampler,
    atomic_json,
    configure_allocator,
    make_bounds,
    make_host_tensor,
)
from seqattn_core.benchmarking.projection import main

__all__ = [
    "MemorySampler",
    "atomic_json",
    "configure_allocator",
    "main",
    "make_bounds",
    "make_host_tensor",
]


if __name__ == "__main__":
    main()
