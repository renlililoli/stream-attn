from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from ..kernels.sol_preprocess import SOL_BLOCK_TOKENS, SOL_HEAD_DIM
from ..plan import AttentionPlan, estimate_workspace_bytes


@dataclass(frozen=True)
class SolStreamingPlan:
    attention: AttentionPlan
    route_block_tokens: int
    max_kv_blocks: int
    max_q_blocks: int
    dense_workspace_bytes: int
    sparse_workspace_bytes: int
    estimated_workspace_bytes: int


def _validate_exact_prefix_tokens(
    exact_prefix_tokens: tuple[int, ...],
    bounds: list[int],
) -> None:
    if len(exact_prefix_tokens) != len(bounds) - 1:
        raise ValueError("exact_prefix_tokens must contain one value per packed segment")
    for index, (prefix, start, stop) in enumerate(
        zip(exact_prefix_tokens, bounds[:-1], bounds[1:])
    ):
        if isinstance(prefix, bool) or not isinstance(prefix, int):
            raise TypeError(f"exact_prefix_tokens[{index}] must be an integer")
        if not 0 <= prefix <= stop - start:
            raise ValueError(f"exact_prefix_tokens[{index}] exceeds its packed segment")


def _aligned_chunk(tokens: int, maximum: int, name: str) -> int:
    if maximum <= SOL_BLOCK_TOKENS:
        return maximum
    aligned = min(tokens, maximum)
    aligned -= aligned % SOL_BLOCK_TOKENS
    if aligned < SOL_BLOCK_TOKENS:
        raise ValueError(f"{name} must provide capacity for one 64-token Sol route block")
    return aligned


def _sparse_workspace_bytes(plan: AttentionPlan, q_chunk_tokens: int) -> int:
    element_size = torch.empty((), dtype=plan.dtype).element_size()
    max_kv_blocks = math.ceil(plan.max_kv_tokens / SOL_BLOCK_TOKENS)
    max_q_blocks = math.ceil(q_chunk_tokens / SOL_BLOCK_TOKENS)
    summaries = 2 * max_kv_blocks * plan.q_heads * plan.head_dim * element_size
    k_statistics = 2 * plan.q_heads * plan.head_dim * 4
    thresholds = max_q_blocks * plan.q_heads * 4
    route_counts = 2 * 8
    return summaries + k_statistics + thresholds + route_counts


def _dense_workspace_bytes(
    plan: AttentionPlan,
    *,
    q_chunk_tokens: int,
    kv_chunk_tokens: int,
) -> int:
    return estimate_workspace_bytes(
        q_tokens=q_chunk_tokens,
        kv_tokens=kv_chunk_tokens,
        q_heads=plan.q_heads,
        kv_heads=plan.kv_heads,
        head_dim=plan.head_dim,
        dtype=plan.dtype,
        num_kv_buffers=plan.num_kv_buffers,
        num_output_buffers=plan.num_output_buffers,
        output_mode=plan.output_mode,
    )


def build_sol_streaming_plan(plan: AttentionPlan) -> SolStreamingPlan:
    """Derive the strict MiniMax-H3 Sol runtime from one resolved dense plan."""

    if plan.device.type != "cuda":
        raise ValueError("sol_streaming requires a CUDA device")
    if plan.output_mode != "device_consumer":
        raise ValueError("sol_streaming V1 requires device_consumer output mode")
    if plan.dtype != torch.bfloat16:
        raise ValueError("sol_streaming V1 requires torch.bfloat16")
    if plan.head_dim != SOL_HEAD_DIM:
        raise ValueError(f"sol_streaming V1 requires head_dim={SOL_HEAD_DIM}")
    if plan.q_heads != plan.kv_heads:
        raise ValueError("sol_streaming V1 requires equal Q and K/V head counts")
    if plan.backend not in {"auto", "triton"}:
        raise ValueError("sol_streaming requires the Triton backend")

    q_chunk = _aligned_chunk(plan.q_chunk_tokens, plan.max_q_tokens, "q_chunk_tokens")
    kv_chunk = _aligned_chunk(plan.kv_chunk_tokens, plan.max_kv_tokens, "kv_chunk_tokens")
    minimum_q = min(SOL_BLOCK_TOKENS, plan.max_q_tokens)
    minimum_kv = min(SOL_BLOCK_TOKENS, plan.max_kv_tokens)
    budget = plan.workspace_budget_bytes
    while True:
        dense_bytes = _dense_workspace_bytes(
            plan,
            q_chunk_tokens=q_chunk,
            kv_chunk_tokens=kv_chunk,
        )
        sparse_bytes = _sparse_workspace_bytes(plan, q_chunk)
        total_bytes = dense_bytes + sparse_bytes
        if budget is None or total_bytes <= budget:
            break
        if q_chunk > minimum_q:
            q_chunk = max(minimum_q, q_chunk - SOL_BLOCK_TOKENS)
        elif kv_chunk > minimum_kv:
            kv_chunk = max(minimum_kv, kv_chunk - SOL_BLOCK_TOKENS)
        else:
            raise ValueError(
                "workspace budget is too small for sol_streaming: "
                f"need at least {total_bytes / 2**20:.1f} MiB"
            )

    runtime_attention = replace(
        plan,
        q_chunk_tokens=q_chunk,
        kv_chunk_tokens=kv_chunk,
        estimated_workspace_bytes=total_bytes,
    )
    return SolStreamingPlan(
        attention=runtime_attention,
        route_block_tokens=SOL_BLOCK_TOKENS,
        max_kv_blocks=math.ceil(plan.max_kv_tokens / SOL_BLOCK_TOKENS),
        max_q_blocks=math.ceil(q_chunk / SOL_BLOCK_TOKENS),
        dense_workspace_bytes=dense_bytes,
        sparse_workspace_bytes=sparse_bytes,
        estimated_workspace_bytes=total_bytes,
    )


__all__ = ["SolStreamingPlan", "build_sol_streaming_plan"]
