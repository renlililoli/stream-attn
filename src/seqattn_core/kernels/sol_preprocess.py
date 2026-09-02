from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts
    triton = None
    tl = None


SOL_BLOCK_TOKENS = 64
SOL_HEAD_DIM = 128


if triton is not None:
    _BLOCK_TOKENS = tl.constexpr(SOL_BLOCK_TOKENS)
    _HEAD_DIM = tl.constexpr(SOL_HEAD_DIM)
    _STATS_GROUP = tl.constexpr(32)

    @triton.jit
    def _summarize_kv_kernel(
        k_ptr,
        v_ptr,
        k_centroid_ptr,
        v_sum_ptr,
        kv_tokens,
        summary_block_offset,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_kcb,
        stride_kch,
        stride_kcd,
        stride_vsb,
        stride_vsh,
        stride_vsd,
        HEADS: tl.constexpr,
    ):
        block = tl.program_id(0)
        head = tl.program_id(1)
        token_offsets = block * _BLOCK_TOKENS + tl.arange(0, _BLOCK_TOKENS)
        dim_offsets = tl.arange(0, _HEAD_DIM)
        token_mask = token_offsets < kv_tokens
        k_offsets = (
            token_offsets[:, None] * stride_kt + head * stride_kh + dim_offsets[None, :] * stride_kd
        )
        v_offsets = (
            token_offsets[:, None] * stride_vt + head * stride_vh + dim_offsets[None, :] * stride_vd
        )
        k = tl.load(k_ptr + k_offsets, mask=token_mask[:, None], other=0.0)
        v = tl.load(v_ptr + v_offsets, mask=token_mask[:, None], other=0.0)
        block_tokens = tl.minimum(_BLOCK_TOKENS, kv_tokens - block * _BLOCK_TOKENS)
        summary_block = summary_block_offset + block
        k_summary_offsets = (
            summary_block * stride_kcb + head * stride_kch + dim_offsets * stride_kcd
        )
        v_summary_offsets = (
            summary_block * stride_vsb + head * stride_vsh + dim_offsets * stride_vsd
        )
        tl.store(
            k_centroid_ptr + k_summary_offsets,
            tl.sum(k.to(tl.float32), axis=0) / block_tokens.to(tl.float32),
        )
        tl.store(v_sum_ptr + v_summary_offsets, tl.sum(v.to(tl.float32), axis=0))

    @triton.jit
    def _k_diag_stats_kernel(
        k_centroid_ptr,
        mean_ptr,
        variance_ptr,
        stride_kcb,
        stride_kch,
        stride_kcd,
        stride_mh,
        stride_md,
        stride_vh,
        stride_vd,
        NUM_BLOCKS: tl.constexpr,
    ):
        head = tl.program_id(0)
        dim_offsets = tl.arange(0, _HEAD_DIM)
        block_offsets = tl.arange(0, _STATS_GROUP)
        total = tl.zeros((_HEAD_DIM,), dtype=tl.float32)
        total_sq = tl.zeros((_HEAD_DIM,), dtype=tl.float32)
        for start in range(0, NUM_BLOCKS, _STATS_GROUP):
            blocks = start + block_offsets
            valid = blocks < NUM_BLOCKS
            offsets = (
                blocks[:, None] * stride_kcb + head * stride_kch + dim_offsets[None, :] * stride_kcd
            )
            values = tl.load(
                k_centroid_ptr + offsets,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            total += tl.sum(values, axis=0)
            total_sq += tl.sum(values * values, axis=0)
        mean = total / NUM_BLOCKS
        variance = tl.maximum(total_sq / NUM_BLOCKS - mean * mean, 0.0)
        tl.store(mean_ptr + head * stride_mh + dim_offsets * stride_md, mean)
        tl.store(variance_ptr + head * stride_vh + dim_offsets * stride_vd, variance)

    @triton.jit
    def _q_diag_threshold_kernel(
        q_ptr,
        mean_ptr,
        variance_ptr,
        threshold_ptr,
        q_tokens,
        scale_log2,
        tau,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_mh,
        stride_md,
        stride_vh,
        stride_vd,
        stride_tb,
        stride_th,
    ):
        q_block = tl.program_id(0)
        head = tl.program_id(1)
        token_offsets = q_block * _BLOCK_TOKENS + tl.arange(0, _BLOCK_TOKENS)
        dim_offsets = tl.arange(0, _HEAD_DIM)
        token_mask = token_offsets < q_tokens
        q_offsets = (
            token_offsets[:, None] * stride_qt + head * stride_qh + dim_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=token_mask[:, None], other=0.0)
        q_tokens_in_block = tl.minimum(
            _BLOCK_TOKENS,
            q_tokens - q_block * _BLOCK_TOKENS,
        ).to(tl.float32)
        q_centroid = tl.sum(q.to(tl.float32), axis=0) / q_tokens_in_block
        mean = tl.load(mean_ptr + head * stride_mh + dim_offsets * stride_md)
        variance = tl.load(variance_ptr + head * stride_vh + dim_offsets * stride_vd)
        projected_mean = tl.sum(q_centroid * mean, axis=0) * scale_log2
        projected_variance = (
            tl.sum(q_centroid * q_centroid * variance, axis=0) * scale_log2 * scale_log2
        )
        threshold = projected_mean + tau * tl.sqrt(projected_variance + 1.0e-6)
        tl.store(threshold_ptr + q_block * stride_tb + head * stride_th, threshold)


def summarize_sol_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    k_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    *,
    kv_tokens: int,
    summary_block_offset: int,
) -> int:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    blocks = triton.cdiv(kv_tokens, SOL_BLOCK_TOKENS)
    _summarize_kv_kernel[(blocks, k.shape[1])](
        k,
        v,
        k_centroids,
        value_sums,
        kv_tokens,
        summary_block_offset,
        *k.stride(),
        *v.stride(),
        *k_centroids.stride(),
        *value_sums.stride(),
        HEADS=k.shape[1],
        num_warps=4,
        num_stages=1,
    )
    return blocks


def compute_sol_k_stats(
    k_centroids: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    num_blocks: int,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    _k_diag_stats_kernel[(k_centroids.shape[1],)](
        k_centroids,
        mean,
        variance,
        *k_centroids.stride(),
        *mean.stride(),
        *variance.stride(),
        NUM_BLOCKS=num_blocks,
        num_warps=4,
        num_stages=1,
    )


def compute_sol_thresholds(
    q: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    thresholds: torch.Tensor,
    *,
    q_tokens: int,
    softmax_scale: float,
    tau: float,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    q_blocks = triton.cdiv(q_tokens, SOL_BLOCK_TOKENS)
    _q_diag_threshold_kernel[(q_blocks, q.shape[1])](
        q,
        mean,
        variance,
        thresholds,
        q_tokens,
        softmax_scale * 1.4426950408889634,
        tau,
        *q.stride(),
        *mean.stride(),
        *variance.stride(),
        *thresholds.stride(),
        num_warps=4,
        num_stages=1,
    )


__all__ = [
    "SOL_BLOCK_TOKENS",
    "SOL_HEAD_DIM",
    "compute_sol_k_stats",
    "compute_sol_thresholds",
    "summarize_sol_kv",
]
