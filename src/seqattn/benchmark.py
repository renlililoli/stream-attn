"""Compatibility facade for the streaming attention benchmark."""

from seqattn_core.benchmarking.common import (
    MemorySampler,
    atomic_json,
    configure_allocator,
    make_bounds,
    make_host_tensor,
    process_vram_mib,
)
from seqattn_core.benchmarking.streaming import full_gpu_attention, main

__all__ = [
    "MemorySampler",
    "atomic_json",
    "configure_allocator",
    "full_gpu_attention",
    "main",
    "make_bounds",
    "make_host_tensor",
    "process_vram_mib",
]


if __name__ == "__main__":
    main()
