from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

AttentionEpilogue = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]
DeviceTileOp = Callable[[torch.Tensor, int, int], torch.Tensor]
LeaseFactory = Callable[[], AbstractContextManager]


@dataclass(frozen=True)
class TiledStageOp:
    operation: DeviceTileOp
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


__all__ = ["AttentionEpilogue", "DeviceTileOp", "LeaseFactory", "TiledStageOp"]
