from __future__ import annotations

import torch


def cu_seqlens_from_padding_mask(mask: torch.Tensor, *, name: str = "padding_mask") -> torch.Tensor:
    """Convert a prefix-valid boolean padding mask into packed CPU boundaries."""

    if mask.device.type != "cpu" or mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a two-dimensional CPU boolean tensor")
    if mask.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch row")
    if mask.shape[1] > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError(f"{name} rows must contain a valid prefix followed by padding")
    lengths = mask.sum(dim=1, dtype=torch.int32)
    boundaries = torch.empty(lengths.numel() + 1, dtype=torch.int32)
    boundaries[0] = 0
    torch.cumsum(lengths, dim=0, out=boundaries[1:])
    return boundaries


def reject_additive_attention_mask(mask: torch.Tensor | None, *, name: str) -> None:
    if mask is None:
        return
    if mask.ndim >= 2 and mask.dtype != torch.bool:
        raise ValueError(
            f"{name} additive QxK masks are not supported; use packed boundaries or "
            "a prefix-valid boolean padding mask"
        )


__all__ = ["cu_seqlens_from_padding_mask", "reject_additive_attention_mask"]
