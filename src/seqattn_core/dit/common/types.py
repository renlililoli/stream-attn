from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch

AttentionEpilogue = Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]
DeviceTileOp = Callable[[torch.Tensor, int, int], torch.Tensor]
LeaseFactory = Callable[[], AbstractContextManager]
__all__ = ["AttentionEpilogue", "DeviceTileOp", "LeaseFactory"]
