from collections.abc import Callable

import torch

QKVProjector = Callable[
    [torch.Tensor, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
QTileProjector = Callable[[torch.Tensor, torch.Tensor, int, int], None]
KVTileProjector = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    None,
]
OutputProjector = Callable[[torch.Tensor, int, int], torch.Tensor]


__all__ = ["KVTileProjector", "OutputProjector", "QKVProjector", "QTileProjector"]
