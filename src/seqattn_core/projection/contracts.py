from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

QKVProjector = Callable[
    [torch.Tensor, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
QProjector = Callable[[torch.Tensor, int, int], torch.Tensor]
KVProjector = Callable[
    [torch.Tensor, int, int],
    tuple[torch.Tensor, torch.Tensor],
]
QTileProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]
KVTileProjector = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    None,
]
OutputProjector = Callable[[torch.Tensor, int, int], torch.Tensor]
LeaseFactory = Callable[[], AbstractContextManager]


@dataclass(frozen=True)
class CrossProjection:
    project_q: QProjector
    project_kv: KVProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class SelfProjection:
    project_qkv: QKVProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class SelfRecomputeProjection:
    project_q: QTileProjector
    project_kv: KVTileProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


@dataclass(frozen=True)
class CrossRecomputeProjection:
    project_q: QTileProjector
    project_kv: KVTileProjector
    weight_lease: LeaseFactory | None = None

    def context(self) -> AbstractContextManager:
        return nullcontext() if self.weight_lease is None else self.weight_lease()


__all__ = [
    "CrossProjection",
    "CrossRecomputeProjection",
    "KVProjector",
    "KVTileProjector",
    "LeaseFactory",
    "OutputProjector",
    "QKVProjector",
    "QProjector",
    "QTileProjector",
    "SelfProjection",
    "SelfRecomputeProjection",
]
