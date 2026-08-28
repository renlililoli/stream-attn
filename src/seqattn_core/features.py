from __future__ import annotations

from importlib.metadata import entry_points
from types import ModuleType


class FeatureNotInstalledError(ImportError):
    pass


def available_features() -> tuple[str, ...]:
    return tuple(sorted(item.name for item in entry_points(group="seqattn_core.features")))


def load_feature(name: str) -> ModuleType:
    matches = [item for item in entry_points(group="seqattn_core.features") if item.name == name]
    if not matches:
        raise FeatureNotInstalledError(
            f"SeqAttn feature {name!r} is not installed; install its separate distribution"
        )
    if len(matches) != 1:
        raise RuntimeError(f"multiple SeqAttn feature providers are registered for {name!r}")
    feature = matches[0].load()
    if not isinstance(feature, ModuleType):
        raise TypeError(f"SeqAttn feature {name!r} did not resolve to a module")
    return feature


__all__ = ["FeatureNotInstalledError", "available_features", "load_feature"]
