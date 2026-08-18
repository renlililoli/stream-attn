from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import StreamingAttentionConfig


def _align_down(value: int, alignment: int) -> int:
    return value - value % alignment


@dataclass(frozen=True)
class AttentionPlan:
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    max_q_tokens: int
    max_kv_tokens: int
    q_chunk_tokens: int
    kv_chunk_tokens: int
    num_kv_buffers: int
    num_output_buffers: int
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    estimated_workspace_bytes: int
    workspace_budget_bytes: int | None

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads


def estimate_workspace_bytes(
    *,
    q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_kv_buffers: int,
    num_output_buffers: int,
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = q_tokens * q_heads * head_dim * element_size
    # FP32 accumulator plus FP32 running max and normalizer.
    state_bytes = q_tokens * q_heads * (head_dim + 2) * 4
    output_bytes = num_output_buffers * q_tokens * q_heads * head_dim * element_size
    kv_bytes = (
        num_kv_buffers * 2 * kv_tokens * kv_heads * head_dim * element_size
    )
    # Events and stream objects are small; the fixed margin mainly absorbs
    # allocator alignment and Triton launch scratch without overstating usable
    # query capacity.
    fixed_margin = 32 * 2**20
    return q_bytes + state_bytes + output_bytes + kv_bytes + fixed_margin


def build_plan(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
    max_q_tokens: int,
    max_kv_tokens: int,
    config: StreamingAttentionConfig | None = None,
) -> AttentionPlan:
    config = StreamingAttentionConfig() if config is None else config
    config.validate()
    device = torch.device(device)
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    if head_dim <= 0 or head_dim > 256 or head_dim % 8:
        raise ValueError("head_dim must be a multiple of 8 and at most 256")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if max_q_tokens <= 0 or max_kv_tokens <= 0:
        raise ValueError("max_q_tokens and max_kv_tokens must be positive")

    kv_chunk = min(config.kv_chunk_tokens, max_kv_tokens)
    q_chunk = min(config.q_chunk_tokens or max_q_tokens, max_q_tokens)
    q_chunk = max(config.block_m, _align_down(q_chunk, config.block_m))
    q_chunk = min(q_chunk, max_q_tokens)

    if config.workspace_budget_bytes is not None:
        minimum = estimate_workspace_bytes(
            q_tokens=min(config.block_m, max_q_tokens),
            kv_tokens=kv_chunk,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_kv_buffers=config.num_kv_buffers,
            num_output_buffers=config.num_output_buffers,
        )
        if minimum > config.workspace_budget_bytes:
            raise ValueError(
                "workspace budget is too small for one query block and the requested KV buffers: "
                f"need at least {minimum / 2**20:.1f} MiB"
            )
        if config.q_chunk_tokens is None:
            low = min(config.block_m, max_q_tokens)
            high = q_chunk
            while low <= high:
                candidate = _align_down((low + high) // 2, config.block_m)
                candidate = max(candidate, min(config.block_m, max_q_tokens))
                needed = estimate_workspace_bytes(
                    q_tokens=candidate,
                    kv_tokens=kv_chunk,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    num_kv_buffers=config.num_kv_buffers,
                    num_output_buffers=config.num_output_buffers,
                )
                if needed <= config.workspace_budget_bytes:
                    q_chunk = candidate
                    low = candidate + config.block_m
                else:
                    high = candidate - config.block_m

    estimated = estimate_workspace_bytes(
        q_tokens=q_chunk,
        kv_tokens=kv_chunk,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
    )
    if config.workspace_budget_bytes is not None and estimated > config.workspace_budget_bytes:
        raise ValueError(
            f"requested chunks need {estimated / 2**20:.1f} MiB, exceeding the "
            f"{config.workspace_budget_bytes / 2**20:.1f} MiB workspace budget"
        )

    return AttentionPlan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=max_q_tokens,
        max_kv_tokens=max_kv_tokens,
        q_chunk_tokens=q_chunk,
        kv_chunk_tokens=kv_chunk,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
        block_m=config.block_m,
        block_n=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        estimated_workspace_bytes=estimated,
        workspace_budget_bytes=config.workspace_budget_bytes,
    )
