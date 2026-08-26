from __future__ import annotations

import math

import torch


def quantize_int8_per_token_group(
    source: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_tokens: int = 64,
) -> None:
    """Symmetric INT8 quantization per token group and KV head."""

    if source.ndim != 3:
        raise ValueError("source must use [tokens, heads, head_dim] layout")
    if quantized.dtype != torch.int8 or quantized.shape != source.shape:
        raise ValueError("quantized must be an int8 tensor matching source")
    groups = math.ceil(source.shape[0] / group_tokens)
    if scales.dtype != torch.float16 or scales.shape != (groups, source.shape[1]):
        raise ValueError("scales must use float16 [groups, heads] layout")
    for group in range(groups):
        start = group * group_tokens
        stop = min(start + group_tokens, source.shape[0])
        tile = source[start:stop].float()
        scale = tile.abs().amax(dim=(0, 2)).div(127.0)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        scales[group].copy_(scale.to(torch.float16))
        quantized[start:stop].copy_(
            torch.round(tile / scale[None, :, None]).clamp_(-127, 127).to(torch.int8)
        )


def dequantize_int8_per_token_group(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype,
    group_tokens: int = 64,
) -> torch.Tensor:
    if quantized.ndim != 3 or quantized.dtype != torch.int8:
        raise ValueError("quantized must use int8 [tokens, heads, head_dim] layout")
    groups = math.ceil(quantized.shape[0] / group_tokens)
    if scales.shape != (groups, quantized.shape[1]):
        raise ValueError("scale shape does not match quantized data")
    token_scales = scales.repeat_interleave(group_tokens, dim=0)[: quantized.shape[0]]
    return (quantized.float() * token_scales[:, :, None].float()).to(dtype)
