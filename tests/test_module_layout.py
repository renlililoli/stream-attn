from seqattn import (
    HostMemoryPlan,
    MemoryPageSource,
    NvmeQKVStore,
    NvmeQKVWriter,
    PagedAttentionRunner,
    ProjectedAttentionRunner,
    SimulatedNvmeDevice,
    StreamingAttentionRunner,
)
from seqattn.benchmark import MemorySampler as LegacyMemorySampler
from seqattn.benchmark import main as LegacyStreamingBenchmarkMain
from seqattn.benchmarking import MemorySampler as PackagedMemorySampler
from seqattn.benchmarking import ProcessMemorySampler as PackagedProcessMemorySampler
from seqattn.benchmarking.paged import main as PackagedPagedBenchmarkMain
from seqattn.benchmarking.projection import main as PackagedProjectionBenchmarkMain
from seqattn.benchmarking.streaming import main as PackagedStreamingBenchmarkMain
from seqattn.host_memory import HostMemoryPlan as LegacyHostMemoryPlan
from seqattn.nvme import NvmeQKVStore as LegacyNvmeQKVStore
from seqattn.nvme import NvmeQKVWriter as LegacyNvmeQKVWriter
from seqattn.paged import MemoryPageSource as PagedMemoryPageSource
from seqattn.paged import PagedAttentionRunner as PackagedPagedAttentionRunner
from seqattn.paged.simulation import SimulatedNvmeDevice as PackagedSimulatedNvmeDevice
from seqattn.paged_benchmark import ProcessMemorySampler as LegacyProcessMemorySampler
from seqattn.paged_benchmark import main as LegacyPagedBenchmarkMain
from seqattn.paged_runtime import PagedAttentionRunner as LegacyPagedAttentionRunner
from seqattn.paging import MemoryPageSource as LegacyMemoryPageSource
from seqattn.pipeline import ProjectedAttentionRunner as LegacyProjectedAttentionRunner
from seqattn.pipeline_benchmark import main as LegacyProjectionBenchmarkMain
from seqattn.projection import ProjectedAttentionRunner as PackagedProjectedAttentionRunner
from seqattn.runtime import StreamingAttentionRunner as LegacyStreamingAttentionRunner
from seqattn.storage import NvmeQKVStore as PackagedNvmeQKVStore
from seqattn.storage import NvmeQKVWriter as PackagedNvmeQKVWriter
from seqattn.streaming import StreamingAttentionRunner as PackagedStreamingAttentionRunner


def test_legacy_facades_preserve_public_class_identity():
    assert LegacyHostMemoryPlan is HostMemoryPlan
    assert LegacyMemoryPageSource is MemoryPageSource
    assert LegacyPagedAttentionRunner is PagedAttentionRunner
    assert LegacyNvmeQKVStore is NvmeQKVStore
    assert LegacyNvmeQKVWriter is NvmeQKVWriter
    assert LegacyProjectedAttentionRunner is ProjectedAttentionRunner
    assert LegacyStreamingAttentionRunner is StreamingAttentionRunner
    assert LegacyMemorySampler is PackagedMemorySampler
    assert LegacyProcessMemorySampler is PackagedProcessMemorySampler
    assert LegacyStreamingBenchmarkMain is PackagedStreamingBenchmarkMain
    assert LegacyPagedBenchmarkMain is PackagedPagedBenchmarkMain
    assert LegacyProjectionBenchmarkMain is PackagedProjectionBenchmarkMain


def test_new_subpackages_export_the_canonical_implementations():
    assert PagedMemoryPageSource is MemoryPageSource
    assert PackagedPagedAttentionRunner is PagedAttentionRunner
    assert PackagedSimulatedNvmeDevice is SimulatedNvmeDevice
    assert PackagedNvmeQKVStore is NvmeQKVStore
    assert PackagedNvmeQKVWriter is NvmeQKVWriter
    assert PackagedProjectedAttentionRunner is ProjectedAttentionRunner
    assert PackagedStreamingAttentionRunner is StreamingAttentionRunner


def test_core_package_contains_no_compat_facades():
    from pathlib import Path

    import seqattn_core

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
    assert not present, f"compat facades belong in seqattn, not seqattn_core: {sorted(present)}"
