from pathlib import Path

import seqattn_core
from seqattn_core import (
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
from seqattn_core.dit import minimax_h3
from seqattn_core.dit.ltx2 import (
    LTX2AttentionPlans,
    LTX2Config,
    LTX2MaterializedRunner,
    LTX2RecomputeRunner,
    build_ltx2_runner,
    load_ltx2_config,
)
from seqattn_core.dit.minimax_h3 import (
    H3Config,
    H3MaterializedRunner,
    H3RecomputeRunner,
    build_h3_runner,
    load_h3_config,
)
from seqattn_core.dit.wan import (
    WanAttentionPlans,
    WanConfig,
    WanMaterializedRunner,
    WanRecomputeRunner,
    build_wan_runner,
    load_wan_config,
)
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
    assert WanMaterializedRunner.__name__ == "WanMaterializedRunner"
    assert WanRecomputeRunner.__name__ == "WanRecomputeRunner"
    assert LTX2MaterializedRunner.__name__ == "LTX2MaterializedRunner"
    assert LTX2RecomputeRunner.__name__ == "LTX2RecomputeRunner"
    assert LTX2AttentionPlans.__name__ == "LTX2AttentionPlans"
    assert WanAttentionPlans.__name__ == "WanAttentionPlans"
    assert H3Config.__name__ == "H3Config"
    assert WanConfig.__name__ == "WanConfig"
    assert LTX2Config.__name__ == "LTX2Config"
    assert callable(load_h3_config)
    assert callable(load_wan_config)
    assert callable(load_ltx2_config)
    assert callable(build_h3_runner)
    assert callable(build_wan_runner)
    assert callable(build_ltx2_runner)
    assert not hasattr(minimax_h3, "H3TileConfig")
    assert not hasattr(minimax_h3, "load_h3_tile_config")
    for name in (
        "H3Config",
        "H3DiTRunner",
        "H3MaterializedRunner",
        "H3RecomputeRunner",
        "LTX2Config",
        "LTX2MaterializedRunner",
        "LTX2RecomputeRunner",
        "WanConfig",
        "WanMaterializedRunner",
        "WanRecomputeRunner",
    ):
        assert not hasattr(seqattn_core, name)
        assert not hasattr(seqattn_core.dit, name)
    assert not any(name.startswith("MultiGpu") for name in dir(seqattn_core))


def test_dit_integrations_have_no_legacy_root_modules():
    dit_dir = Path(seqattn_core.dit.__file__).parent
    legacy_modules = {
        "config.py",
        "consumer.py",
        "materialized_runner.py",
        "multigpu.py",
        "projection.py",
        "recompute_runner.py",
        "types.py",
        "validation.py",
        "workspace.py",
    }
    present = {name for name in legacy_modules if (dit_dir / name).exists()}
    assert not present, f"model-specific DiT code must not live at dit root: {sorted(present)}"


def test_repository_benchmark_modules_expose_main_functions():
    assert callable(MemorySampler)
    assert callable(ProcessMemorySampler)
    assert callable(streaming_benchmark_main)
    assert callable(paged_benchmark_main)
    assert callable(projection_benchmark_main)


def test_distribution_contains_only_the_core_package():
    source_root = Path(seqattn_core.__file__).parent.parent
    assert not (source_root / "seqattn").exists()
    assert not (source_root / "seqattn_multigpu").exists()


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
