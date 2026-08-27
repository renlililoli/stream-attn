from typing import Protocol

import torch


class QKVProjector(Protocol):
    """Projects one hidden tile on the current runner compute stream.

    The callback must submit all work to the current stream and return fully
    written Q/K/V tensors on the same CUDA device as ``hidden``.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        start: int,
        stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class QTileProjector(Protocol):
    """Writes a complete Q tile on the current runner compute stream."""

    def __call__(
        self,
        hidden: torch.Tensor,
        destination_q: torch.Tensor,
        start: int,
        stop: int,
    ) -> None: ...


class KVTileProjector(Protocol):
    """Writes complete K and V tiles on the current runner compute stream."""

    def __call__(
        self,
        hidden: torch.Tensor,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        start: int,
        stop: int,
    ) -> None: ...


class OutputProjector(Protocol):
    """Projects an attention tile on the current runner compute stream.

    The returned tensor must be fully written on that stream, reside on the
    planned CUDA device, and cover the complete destination range.
    """

    def __call__(
        self,
        attention: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor: ...


__all__ = ["KVTileProjector", "OutputProjector", "QKVProjector", "QTileProjector"]
