from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache import CacheLookup
from ..layout import PageDescriptor, PageReadMetrics


@dataclass
class KVStage:
    k: torch.Tensor
    v: torch.Tensor
    k_scales: torch.Tensor | None
    v_scales: torch.Tensor | None


@dataclass(frozen=True)
class LoadedPage:
    page: PageDescriptor
    stage_index: int
    metrics: PageReadMetrics
    cache: CacheLookup


__all__ = ["KVStage", "LoadedPage"]
