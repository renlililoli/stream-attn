from __future__ import annotations

import torch

from .config import StreamingAttentionConfig
from .plan import AttentionPlan, build_attention_plan
from .stats import StreamingAttentionStats
from .streaming import StreamingAttentionRunner


def _max_segment_length(cu_seqlens: torch.Tensor) -> int:
    return int(torch.diff(cu_seqlens.to(dtype=torch.int64)).max().item())


def streaming_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    config: StreamingAttentionConfig | None = None,
    out: torch.Tensor | None = None,
    stats: StreamingAttentionStats | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """FlashAttention-style varlen API for CPU-backed packed Q/K/V.

    Inputs use ``[total_tokens, heads, head_dim]``.  Unlike FlashAttention,
    tensors remain in host DRAM and the returned output is also CPU-backed.
    """

    if dropout_p != 0:
        raise ValueError("seqattn V1 is inference-only and requires dropout_p=0")
    actual_max_q = _max_segment_length(cu_seqlens_q)
    actual_max_k = _max_segment_length(cu_seqlens_k)
    if max_seqlen_q is not None and actual_max_q > max_seqlen_q:
        raise ValueError("cu_seqlens_q exceeds max_seqlen_q")
    if max_seqlen_k is not None and actual_max_k > max_seqlen_k:
        raise ValueError("cu_seqlens_k exceeds max_seqlen_k")
    config = StreamingAttentionConfig() if config is None else config
    plan = build_attention_plan(
        q_heads=q.shape[1],
        kv_heads=k.shape[1],
        head_dim=q.shape[2],
        dtype=q.dtype,
        device=device,
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        config=config,
    )
    return StreamingAttentionRunner(plan)(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=causal,
        out=out,
        stats=stats,
    )


def streaming_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    config: StreamingAttentionConfig | None = None,
    out: torch.Tensor | None = None,
    stats: StreamingAttentionStats | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Dense batched wrapper using CPU ``[batch, seqlen, heads, dim]`` tensors."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense q, k, and v must use [batch, seqlen, heads, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("dense q/k/v batch and k/v shapes must match")
    batch, q_length, q_heads, head_dim = q.shape
    k_length, kv_heads = k.shape[1], k.shape[2]
    q_packed = q.reshape(batch * q_length, q_heads, head_dim)
    k_packed = k.reshape(batch * k_length, kv_heads, head_dim)
    v_packed = v.reshape(batch * k_length, kv_heads, head_dim)
    cu_q = torch.arange(0, (batch + 1) * q_length, q_length, dtype=torch.int32)
    cu_k = torch.arange(0, (batch + 1) * k_length, k_length, dtype=torch.int32)
    out_packed = None if out is None else out.reshape_as(q_packed)
    result = streaming_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_q,
        cu_k,
        q_length,
        k_length,
        dropout_p,
        softmax_scale,
        causal,
        config=config,
        out=out_packed,
        stats=stats,
        device=device,
    )
    return result.reshape(batch, q_length, q_heads, head_dim)


__all__ = [
    "AttentionPlan",
    "StreamingAttentionRunner",
    "streaming_attn_func",
    "streaming_attn_varlen_func",
]
