from pathlib import Path

import seqattn_core
from seqattn_core import (
    H3MaterializedRunner,
    H3RecomputeRunner,
    MemoryPageSource,
    NvmeQKVStore,
    NvmeQKVWriter,
    PagedAttentionRunner,
    ProjectedAttentionRunner,
    RecomputedAttentionRunner,
    SimulatedNvmeDevice,
    StreamingAttentionRunner,
)
from seqattn_core.benchmarking import MemorySampler, ProcessMemorySampler
from seqattn_core.benchmarking.paged import main as paged_benchmark_main
from seqattn_core.benchmarking.projection import main as projection_benchmark_main
from seqattn_core.benchmarking.streaming import main as streaming_benchmark_main
from seqattn_core.paged import MemoryPageSource as PagedMemoryPageSource
from seqattn_core.paged import PagedAttentionRunner as PackagedPagedAttentionRunner
from seqattn_core.paged.simulation import SimulatedNvmeDevice as PackagedSimulatedNvmeDevice
from seqattn_core.projection import ProjectedAttentionRunner as PackagedProjectedAttentionRunner
from seqattn_core.projection import (
    RecomputedAttentionRunner as PackagedRecomputedAttentionRunner,
)
from seqattn_core.storage import NvmeQKVStore as PackagedNvmeQKVStore
from seqattn_core.storage import NvmeQKVWriter as PackagedNvmeQKVWriter
from seqattn_core.streaming import StreamingAttentionRunner as PackagedStreamingAttentionRunner


def test_subpackages_export_the_canonical_implementations():
    assert PagedMemoryPageSource is MemoryPageSource
    assert PackagedPagedAttentionRunner is PagedAttentionRunner
    assert PackagedSimulatedNvmeDevice is SimulatedNvmeDevice
    assert PackagedNvmeQKVStore is NvmeQKVStore
    assert PackagedNvmeQKVWriter is NvmeQKVWriter
    assert PackagedProjectedAttentionRunner is ProjectedAttentionRunner
    assert PackagedRecomputedAttentionRunner is RecomputedAttentionRunner
    assert PackagedStreamingAttentionRunner is StreamingAttentionRunner
    assert H3MaterializedRunner.__name__ == "H3MaterializedRunner"
    assert H3RecomputeRunner.__name__ == "H3RecomputeRunner"
    assert not hasattr(seqattn_core, "H3DiTRunner")


def test_repository_benchmark_modules_expose_main_functions():
    assert callable(MemorySampler)
    assert callable(ProcessMemorySampler)
    assert callable(streaming_benchmark_main)
    assert callable(paged_benchmark_main)
    assert callable(projection_benchmark_main)


def test_distribution_contains_only_the_core_package():
    source_root = Path(seqattn_core.__file__).parent.parent
    assert not (source_root / "seqattn").exists()


def test_core_package_contains_no_compat_facades():
    compat_facades = {
        "benchmark",
        "cache",
        "host_memory",
        "nvme",
        "paged_benchmark",
        "paged_runtime",
        "paging",
        "pipeline",
        "pipeline_benchmark",
        "runtime",
        "simulated_nvme",
    }
    core_dir = Path(seqattn_core.__file__).parent
    present = {name for name in compat_facades if (core_dir / f"{name}.py").exists()}
    assert not present, f"compatibility facades must not ship: {sorted(present)}"
