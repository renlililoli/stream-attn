from pathlib import Path

import seqattn_core
from seqattn_core import features

import seqattn_multigpu
from seqattn_multigpu import planning, streaming


def test_plugin_exports_multigpu_api_and_core_does_not():
    assert seqattn_multigpu.__version__ == "0.4.0a1"
    assert seqattn_multigpu.MultiGpuH3MaterializedRunner.__name__ == (
        "MultiGpuH3MaterializedRunner"
    )
    assert not any(name.startswith("MultiGpu") for name in dir(seqattn_core))


def test_plugin_separates_planning_from_streaming_execution():
    plugin_dir = Path(seqattn_multigpu.__file__).parent
    assert (plugin_dir / "planning.py").exists()
    assert seqattn_multigpu.MultiGpuAttentionPlan is planning.MultiGpuAttentionPlan
    assert seqattn_multigpu.MultiGpuDeviceSpec is planning.MultiGpuDeviceSpec
    assert seqattn_multigpu.build_multi_gpu_plan is planning.build_multi_gpu_plan
    assert not hasattr(streaming, "MultiGpuAttentionPlan")
    assert not hasattr(streaming, "MultiGpuDeviceSpec")
    assert not hasattr(streaming, "build_multi_gpu_plan")


def test_plugin_is_discoverable_through_feature_entry_point():
    assert "multigpu" in features.available_features()
    assert features.load_feature("multigpu") is seqattn_multigpu
