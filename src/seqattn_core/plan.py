from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import StreamingAttentionConfig, configured_backend_name
from .kernels.profiles import PORTABLE_KERNEL, resolve_builtin_kernel_profile

_DEFAULT_KV_CHUNK_TOKENS = 8192


def _align_down(value: int, alignment: int) -> int:
    return value - value % alignment


@dataclass(frozen=True)
class _KernelLaunch:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


def _resolve_kernel_launch(
    config: StreamingAttentionConfig,
    *,
    device: torch.device,
    head_dim: int,
    dtype: torch.dtype,
) -> _KernelLaunch:
    values = (config.block_m, config.block_n, config.num_warps, config.num_stages)
    base = (
        PORTABLE_KERNEL
        if any(value is not None for value in values)
        else resolve_builtin_kernel_profile(device=device, head_dim=head_dim, dtype=dtype)
    )
    block_m, block_n, num_warps, num_stages = (
        default if value is None else value for value, default in zip(values, base)
    )
    return _KernelLaunch(
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@dataclass(frozen=True)
class AttentionPlan:
    """Fully resolved, immutable inputs for one streaming attention runner."""

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
    output_mode: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    backend: str
    require_pinned: bool
    pin_output: bool
    enable_nvtx: bool
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
    output_mode: str = "host",
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = q_tokens * q_heads * head_dim * element_size
    state_bytes = q_tokens * q_heads * (head_dim + 2) * 4
    output_bytes = (
        num_output_buffers * q_tokens * q_heads * head_dim * element_size
        if output_mode == "host"
        else 0
    )
    kv_bytes = num_kv_buffers * 2 * kv_tokens * kv_heads * head_dim * element_size
    return q_bytes + state_bytes + output_bytes + kv_bytes + 32 * 2**20


def _normalize_chunk(requested: int, maximum: int, alignment: int) -> int:
    minimum = min(alignment, maximum)
    return min(maximum, max(minimum, _align_down(min(requested, maximum), alignment)))


def _largest_q_chunk_that_fits(
    *,
    max_q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_m: int,
    workspace_budget_bytes: int | None,
    num_kv_buffers: int,
    num_output_buffers: int,
    output_mode: str,
) -> int | None:
    minimum_q = min(block_m, max_q_tokens)
    budget = workspace_budget_bytes
    if budget is None:
        return max_q_tokens

    def required(q_tokens: int) -> int:
        return estimate_workspace_bytes(
            q_tokens=q_tokens,
            kv_tokens=kv_tokens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            num_kv_buffers=num_kv_buffers,
            num_output_buffers=num_output_buffers,
            output_mode=output_mode,
        )

    if required(minimum_q) > budget:
        return None
    low, high, best = minimum_q, max_q_tokens, minimum_q
    while low <= high:
        candidate = max(minimum_q, _align_down((low + high) // 2, block_m))
        if required(candidate) <= budget:
            best = candidate
            low = candidate + block_m
        else:
            high = candidate - block_m
    return min(best, max_q_tokens)


def build_attention_plan(
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
    """Resolve shape, workspace and execution policy without performance guessing."""

    config = StreamingAttentionConfig() if config is None else config
    config.validate()
    device = torch.device(device)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads must be a positive multiple of kv_heads")
    if head_dim <= 0 or head_dim > 256 or head_dim % 8:
        raise ValueError("head_dim must be a multiple of 8 and at most 256")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if max_q_tokens <= 0 or max_kv_tokens <= 0:
        raise ValueError("max_q_tokens and max_kv_tokens must be positive")

    kernel = _resolve_kernel_launch(config, device=device, head_dim=head_dim, dtype=dtype)
    requested_kv = (
        _DEFAULT_KV_CHUNK_TOKENS if config.kv_chunk_tokens is None else config.kv_chunk_tokens
    )
    kv_chunk = _normalize_chunk(requested_kv, max_kv_tokens, kernel.block_n)
    if config.q_chunk_tokens is None:
        q_chunk = _largest_q_chunk_that_fits(
            max_q_tokens=max_q_tokens,
            kv_tokens=kv_chunk,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            block_m=kernel.block_m,
            workspace_budget_bytes=config.workspace_budget_bytes,
            num_kv_buffers=config.num_kv_buffers,
            num_output_buffers=config.num_output_buffers,
            output_mode=config.output_mode,
        )
        if q_chunk is None:
            minimum = estimate_workspace_bytes(
                q_tokens=min(kernel.block_m, max_q_tokens),
                kv_tokens=kv_chunk,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                num_kv_buffers=config.num_kv_buffers,
                num_output_buffers=config.num_output_buffers,
                output_mode=config.output_mode,
            )
            raise ValueError(
                "workspace budget is too small for one query block and the requested KV "
                f"buffers: need at least {minimum / 2**20:.1f} MiB"
            )
    else:
        q_chunk = _normalize_chunk(config.q_chunk_tokens, max_q_tokens, kernel.block_m)

    estimated = estimate_workspace_bytes(
        q_tokens=q_chunk,
        kv_tokens=kv_chunk,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_kv_buffers=config.num_kv_buffers,
        num_output_buffers=config.num_output_buffers,
        output_mode=config.output_mode,
    )
    budget = config.workspace_budget_bytes
    if budget is not None and estimated > budget:
        raise ValueError(
            f"requested chunks need {estimated / 2**20:.1f} MiB, exceeding the "
            f"{budget / 2**20:.1f} MiB workspace budget"
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
        output_mode=config.output_mode,
        block_m=kernel.block_m,
        block_n=kernel.block_n,
        num_warps=kernel.num_warps,
        num_stages=kernel.num_stages,
        backend=configured_backend_name(config.backend),
        require_pinned=config.require_pinned,
        pin_output=config.pin_output,
        enable_nvtx=config.enable_nvtx,
        estimated_workspace_bytes=estimated,
        workspace_budget_bytes=budget,
    )


__all__ = ["AttentionPlan", "build_attention_plan", "estimate_workspace_bytes"]
