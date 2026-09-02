from __future__ import annotations

import torch

from .sol_preprocess import SOL_BLOCK_TOKENS, SOL_HEAD_DIM

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts
    triton = None
    tl = None

if triton is not None:
    _BLOCK_TOKENS = tl.constexpr(SOL_BLOCK_TOKENS)
    _HEAD_DIM = tl.constexpr(SOL_HEAD_DIM)
    _ROUTE_GROUP = tl.constexpr(16)

    @triton.jit
    def _sol_streaming_update_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        k_centroid_ptr,
        v_sum_ptr,
        threshold_ptr,
        max_ptr,
        sum_ptr,
        acc_ptr,
        route_counts_ptr,
        q_tokens,
        kv_tokens,
        q_block_offset,
        kv_block_offset,
        segment_blocks,
        exact_prefix_blocks,
        scale_log2,
        stride_qt,
        stride_qh,
        stride_qd,
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
        stride_tb,
        stride_th,
        stride_st,
        stride_sh,
        stride_at,
        stride_ah,
        stride_ad,
        TILE_BLOCKS: tl.constexpr,
        INITIALIZE: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        head = tl.program_id(1)
        token_offsets = q_block * _BLOCK_TOKENS + tl.arange(0, _BLOCK_TOKENS)
        dim_offsets = tl.arange(0, _HEAD_DIM)
        q_mask = token_offsets < q_tokens
        q_offsets = (
            token_offsets[:, None] * stride_qt + head * stride_qh + dim_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=q_mask[:, None], other=0.0)
        q_tokens_in_block = tl.minimum(
            _BLOCK_TOKENS,
            q_tokens - q_block * _BLOCK_TOKENS,
        ).to(tl.float32)
        local_q_block = q_block_offset + q_block
        threshold = tl.load(threshold_ptr + q_block * stride_tb + head * stride_th)

        state_offsets = token_offsets * stride_st + head * stride_sh
        acc_offsets = (
            token_offsets[:, None] * stride_at + head * stride_ah + dim_offsets[None, :] * stride_ad
        )
        if INITIALIZE:
            running_max = tl.full((_BLOCK_TOKENS,), -float("inf"), tl.float32)
            running_sum = tl.zeros((_BLOCK_TOKENS,), tl.float32)
            accumulator = tl.zeros((_BLOCK_TOKENS, _HEAD_DIM), tl.float32)
        else:
            running_max = tl.load(max_ptr + state_offsets, mask=q_mask, other=-float("inf"))
            running_sum = tl.load(sum_ptr + state_offsets, mask=q_mask, other=0.0)
            accumulator = tl.load(
                acc_ptr + acc_offsets,
                mask=q_mask[:, None],
                other=0.0,
            ).to(tl.float32)

        group_offsets = tl.arange(0, _ROUTE_GROUP)
        dense_query_block = local_q_block < exact_prefix_blocks
        for group_start in range(0, TILE_BLOCKS, _ROUTE_GROUP):
            tile_blocks = group_start + group_offsets
            local_blocks = kv_block_offset + tile_blocks
            valid_blocks = (tile_blocks < TILE_BLOCKS) & (local_blocks < segment_blocks)
            summary_k_offsets = (
                local_blocks[:, None] * stride_kcb
                + head * stride_kch
                + dim_offsets[None, :] * stride_kcd
            )
            summary_v_offsets = (
                local_blocks[:, None] * stride_vsb
                + head * stride_vsh
                + dim_offsets[None, :] * stride_vsd
            )
            k_centroids = tl.load(
                k_centroid_ptr + summary_k_offsets,
                mask=valid_blocks[:, None],
                other=0.0,
            )
            value_sums = tl.load(
                v_sum_ptr + summary_v_offsets,
                mask=valid_blocks[:, None],
                other=0.0,
            )
            centroid_scores = tl.dot(q, tl.trans(k_centroids)).to(tl.float32) * scale_log2
            route_scores = tl.sum(centroid_scores, axis=0) / q_tokens_in_block
            exact = (
                (route_scores > threshold)
                | (tl.abs(local_q_block - local_blocks) <= 1)
                | (local_blocks < exact_prefix_blocks)
                | dense_query_block
            ) & valid_blocks
            approximate = valid_blocks & ~exact

            has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
            approximate_scores = tl.where(
                approximate[None, :],
                centroid_scores,
                -float("inf"),
            )
            safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
            candidate_max = tl.maximum(running_max, tl.max(safe_scores, axis=1))
            merged_max = tl.where(has_approximate, candidate_max, running_max)
            alpha = tl.math.exp2(tl.where(has_approximate, running_max - merged_max, 0.0))
            probabilities = tl.math.exp2(
                safe_scores - tl.where(has_approximate, merged_max, 0.0)[:, None]
            )
            probabilities = tl.where(
                has_approximate & approximate[None, :],
                probabilities,
                0.0,
            )
            block_lengths = tl.minimum(
                _BLOCK_TOKENS,
                tl.maximum(0, segment_blocks * _BLOCK_TOKENS - local_blocks * _BLOCK_TOKENS),
            ).to(tl.float32)
            # The true segment tail can be shorter than segment_blocks * 64. The
            # final block length is corrected below from the resident tile tail.
            final_segment_block = local_blocks == segment_blocks - 1
            segment_tail = (kv_tokens + kv_block_offset * _BLOCK_TOKENS) - (
                segment_blocks - 1
            ) * _BLOCK_TOKENS
            block_lengths = tl.where(final_segment_block, segment_tail, block_lengths)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(value_sums.dtype),
                value_sums,
            )
            running_sum = running_sum * alpha + tl.sum(
                probabilities * block_lengths[None, :],
                axis=1,
            )
            running_max = merged_max

            exact_offsets = tl.where(exact, group_offsets, _ROUTE_GROUP)
            num_exact = tl.sum(exact.to(tl.int32), axis=0)
            for _ in range(num_exact):
                offset = tl.min(exact_offsets)
                tile_block = group_start + offset
                exact_offsets = tl.where(
                    group_offsets == offset,
                    _ROUTE_GROUP,
                    exact_offsets,
                )
                kv_token_offsets = tile_block * _BLOCK_TOKENS + tl.arange(0, _BLOCK_TOKENS)
                kv_mask = kv_token_offsets < kv_tokens
                k_offsets = (
                    kv_token_offsets[:, None] * stride_kt
                    + head * stride_kh
                    + dim_offsets[None, :] * stride_kd
                )
                v_offsets = (
                    kv_token_offsets[:, None] * stride_vt
                    + head * stride_vh
                    + dim_offsets[None, :] * stride_vd
                )
                k = tl.load(k_ptr + k_offsets, mask=kv_mask[:, None], other=0.0)
                logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
                valid = q_mask[:, None] & kv_mask[None, :]
                logits = tl.where(valid, logits, -float("inf"))
                tile_max = tl.max(logits, axis=1)
                new_max = tl.maximum(running_max, tile_max)
                row_valid = new_max != -float("inf")
                exact_alpha = tl.where(
                    row_valid,
                    tl.math.exp2(running_max - new_max),
                    1.0,
                )
                exact_probabilities = tl.where(
                    valid,
                    tl.math.exp2(logits - new_max[:, None]),
                    0.0,
                )
                v = tl.load(v_ptr + v_offsets, mask=kv_mask[:, None], other=0.0)
                accumulator = accumulator * exact_alpha[:, None] + tl.dot(
                    exact_probabilities.to(v.dtype),
                    v,
                )
                running_sum = running_sum * exact_alpha + tl.sum(
                    exact_probabilities,
                    axis=1,
                )
                running_max = tl.where(row_valid, new_max, running_max)

            tl.atomic_add(route_counts_ptr, tl.sum(exact.to(tl.int64), axis=0))
            tl.atomic_add(
                route_counts_ptr + 1,
                tl.sum(approximate.to(tl.int64), axis=0),
            )

        tl.store(max_ptr + state_offsets, running_max, mask=q_mask)
        tl.store(sum_ptr + state_offsets, running_sum, mask=q_mask)
        tl.store(acc_ptr + acc_offsets, accumulator, mask=q_mask[:, None])


def update_sol_attention_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    thresholds: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    accumulator: torch.Tensor,
    route_counts: torch.Tensor,
    *,
    q_tokens: int,
    kv_tokens: int,
    q_block_offset: int,
    kv_block_offset: int,
    segment_tokens: int,
    exact_prefix_tokens: int,
    softmax_scale: float,
    initialize: bool,
) -> None:
    if triton is None:
        raise RuntimeError("the Triton backend is not installed")
    q_blocks = triton.cdiv(q_tokens, SOL_BLOCK_TOKENS)
    tile_blocks = triton.cdiv(kv_tokens, SOL_BLOCK_TOKENS)
    segment_blocks = triton.cdiv(segment_tokens, SOL_BLOCK_TOKENS)
    exact_prefix_blocks = triton.cdiv(exact_prefix_tokens, SOL_BLOCK_TOKENS)
    _sol_streaming_update_kernel[(q_blocks, q.shape[1])](
        q,
        k,
        v,
        k_centroids,
        value_sums,
        thresholds,
        running_max,
        running_sum,
        accumulator,
        route_counts,
        q_tokens,
        kv_tokens,
        q_block_offset,
        kv_block_offset,
        segment_blocks,
        exact_prefix_blocks,
        softmax_scale * 1.4426950408889634,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *k_centroids.stride(),
        *value_sums.stride(),
        *thresholds.stride(),
        *running_max.stride(),
        *accumulator.stride(),
        TILE_BLOCKS=tile_blocks,
        INITIALIZE=initialize,
        num_warps=8,
        num_stages=1,
    )


__all__ = [
    "update_sol_attention_state",
]
