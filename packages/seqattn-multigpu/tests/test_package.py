import seqattn_core
from seqattn_core import features

import seqattn_multigpu


def test_plugin_exports_multigpu_api_and_core_does_not():
    assert seqattn_multigpu.__version__ == "0.3.0a4"
    assert seqattn_multigpu.MultiGpuH3MaterializedRunner.__name__ == (
        "MultiGpuH3MaterializedRunner"
    )
    assert not any(name.startswith("MultiGpu") for name in dir(seqattn_core))


def test_plugin_is_discoverable_through_feature_entry_point():
    assert "multigpu" in features.available_features()
    assert features.load_feature("multigpu") is seqattn_multigpu
