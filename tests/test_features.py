from types import SimpleNamespace

import pytest

from seqattn_core import features


def test_missing_feature_has_explicit_error(monkeypatch):
    monkeypatch.setattr(features, "entry_points", lambda **_kwargs: ())

    assert features.available_features() == ()
    with pytest.raises(features.FeatureNotInstalledError, match="not installed"):
        features.load_feature("multigpu")


def test_feature_loader_uses_registered_entry_point(monkeypatch):
    module = __import__("types")
    entry_point = SimpleNamespace(name="multigpu", load=lambda: module)
    monkeypatch.setattr(features, "entry_points", lambda **_kwargs: (entry_point,))

    assert features.available_features() == ("multigpu",)
    assert features.load_feature("multigpu") is module
