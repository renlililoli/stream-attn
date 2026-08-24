from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only development hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _initialize_split_state_kernel(
        partial_output_ptr,
        partial_lse_ptr,
        state_output_ptr,
        state_lse_ptr,
        total_values,
        SEQ_LEN: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value_mask = offsets < total_values
        rows = offsets // HEAD_DIM
        dims = offsets - rows * HEAD_DIM
        batch = rows // (SEQ_LEN * HEADS)
        row_in_batch = rows - batch * SEQ_LEN * HEADS
        tokens = row_in_batch // HEADS
        heads = row_in_batch - tokens * HEADS
        lse_offsets = (batch * HEADS + heads) * SEQ_LEN + tokens

        partial = tl.load(partial_output_ptr + offsets, mask=value_mask, other=0.0)
        tl.store(state_output_ptr + offsets, partial.to(tl.float32), mask=value_mask)
        lse_mask = value_mask & (dims == 0)
        local_lse = tl.load(partial_lse_ptr + lse_offsets, mask=lse_mask, other=-float("inf"))
        tl.store(state_lse_ptr + rows, local_lse, mask=lse_mask)

    @triton.jit
    def _merge_split_state_kernel(
        partial_output_ptr,
        partial_lse_ptr,
        state_output_ptr,
        state_lse_ptr,
        total_values,
        SEQ_LEN: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value_mask = offsets < total_values
        rows = offsets // HEAD_DIM
        dims = offsets - rows * HEAD_DIM
        batch = rows // (SEQ_LEN * HEADS)
        row_in_batch = rows - batch * SEQ_LEN * HEADS
        tokens = row_in_batch // HEADS
        heads = row_in_batch - tokens * HEADS
        lse_offsets = (batch * HEADS + heads) * SEQ_LEN + tokens

        previous_lse = tl.load(state_lse_ptr + rows, mask=value_mask, other=-float("inf"))
        local_lse = tl.load(partial_lse_ptr + lse_offsets, mask=value_mask, other=-float("inf"))
        merged_max = tl.maximum(previous_lse, local_lse)
        previous_term = tl.where(
            previous_lse == -float("inf"),
            0.0,
            tl.exp(previous_lse - merged_max),
        )
        local_term = tl.where(
            local_lse == -float("inf"),
            0.0,
            tl.exp(local_lse - merged_max),
        )
        denominator = previous_term + local_term
        merged_lse = tl.where(
            denominator > 0.0,
            merged_max + tl.log(denominator),
            -float("inf"),
        )
        previous_weight = tl.where(denominator > 0.0, previous_term / denominator, 0.0)
        local_weight = tl.where(denominator > 0.0, local_term / denominator, 0.0)
        previous_output = tl.load(state_output_ptr + offsets, mask=value_mask, other=0.0)
        partial_output = tl.load(partial_output_ptr + offsets, mask=value_mask, other=0.0)
        merged_output = previous_weight * previous_output + local_weight * partial_output
        tl.store(state_output_ptr + offsets, merged_output, mask=value_mask)
        tl.store(state_lse_ptr + rows, merged_lse, mask=value_mask & (dims == 0))


def _validate_split_state(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
) -> tuple[int, int, int, int]:
    if triton is None:
        raise RuntimeError("split attention state combine requires Triton")
    if partial_output.ndim != 4:
        raise ValueError("partial_output must have shape [batch, tokens, heads, head_dim]")
    batch, tokens, heads, head_dim = partial_output.shape
    if partial_lse.shape != (batch, heads, tokens):
        raise ValueError("partial_lse must have shape [batch, heads, tokens]")
    if state_output.shape != partial_output.shape or state_output.dtype != torch.float32:
        raise ValueError("state_output must be FP32 and match partial_output shape")
    if state_lse.shape != (batch, tokens, heads) or state_lse.dtype != torch.float32:
        raise ValueError("state_lse must be FP32 with shape [batch, tokens, heads]")
    tensors = (partial_output, partial_lse, state_output, state_lse)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("split attention state tensors must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("split attention state tensors must be contiguous")
    if partial_lse.dtype != torch.float32:
        raise ValueError("partial_lse must use FP32")
    if partial_output.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("partial_output must use FP16, BF16, or FP32")
    return batch, tokens, heads, head_dim


def initialize_split_attention_state(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
) -> None:
    """Initialize normalized FP32 attention state from one K/V partition."""

    _, tokens, heads, head_dim = _validate_split_state(
        partial_output, partial_lse, state_output, state_lse
    )
    total_values = partial_output.numel()
    block = 256
    _initialize_split_state_kernel[(triton.cdiv(total_values, block),)](
        partial_output,
        partial_lse,
        state_output,
        state_lse,
        total_values,
        SEQ_LEN=tokens,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK=block,
        num_warps=4,
    )


def merge_split_attention_state(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    state_output: torch.Tensor,
    state_lse: torch.Tensor,
) -> None:
    """Merge one normalized partial attention result using its FP32 LSE."""

    _, tokens, heads, head_dim = _validate_split_state(
        partial_output, partial_lse, state_output, state_lse
    )
    total_values = partial_output.numel()
    block = 256
    _merge_split_state_kernel[(triton.cdiv(total_values, block),)](
        partial_output,
        partial_lse,
        state_output,
        state_lse,
        total_values,
        SEQ_LEN=tokens,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK=block,
        num_warps=4,
    )


__all__ = ["initialize_split_attention_state", "merge_split_attention_state"]
