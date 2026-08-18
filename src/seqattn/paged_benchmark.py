"""Compatibility facade for the paged attention benchmark."""

from seqattn_core.benchmarking.common import (
    ProcessMemorySampler,
    atomic_json,
    make_bounds,
    process_rss_bytes,
    process_vram_bytes,
)
from seqattn_core.benchmarking.paged import main, make_tensor, stream_store

__all__ = [
    "ProcessMemorySampler",
    "atomic_json",
    "main",
    "make_bounds",
    "make_tensor",
    "process_rss_bytes",
    "process_vram_bytes",
    "stream_store",
]


if __name__ == "__main__":
    main()
