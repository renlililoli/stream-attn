from __future__ import annotations

import torch

from ..plan import AttentionPlan


def validate_projection_hidden(
    hidden_cpu: torch.Tensor,
    *,
    plan: AttentionPlan,
    require_pinned: bool,
    hidden_features: int | None = None,
    max_tokens: int | None = None,
    name: str = "hidden_cpu",
) -> None:
    if hidden_cpu.device.type != "cpu" or hidden_cpu.ndim != 2:
        raise ValueError(f"{name} must use CPU [tokens, hidden_features] layout")
    if hidden_features is not None and hidden_cpu.shape[1] != hidden_features:
        raise ValueError(f"{name} feature size does not match the runner")
    if not hidden_cpu.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if hidden_cpu.dtype != plan.dtype:
        raise ValueError(f"{name} dtype must match the attention plan")
    if require_pinned and not hidden_cpu.is_pinned():
        raise ValueError(f"asynchronous projection requires pinned {name}")
    tokens = hidden_cpu.shape[0]
    if max_tokens is None:
        max_tokens = min(plan.max_q_tokens, plan.max_kv_tokens)
    if tokens > max_tokens:
        raise ValueError(f"{name} token count exceeds the runner plan")


def validate_projected_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tokens: int,
    plan: AttentionPlan,
) -> None:
    expected_q = (tokens, plan.q_heads, plan.head_dim)
    expected_kv = (tokens, plan.kv_heads, plan.head_dim)
    if q.shape != expected_q:
        raise ValueError(f"project_qkv returned q shape {tuple(q.shape)}, expected {expected_q}")
    if k.shape != expected_kv or v.shape != expected_kv:
        raise ValueError(
            "project_qkv returned invalid k/v shapes: "
            f"{tuple(k.shape)}, {tuple(v.shape)}, expected {expected_kv}"
        )
    if any(tensor.device != plan.device for tensor in (q, k, v)):
        raise ValueError(f"project_qkv must return tensors on {plan.device}")
    if any(tensor.dtype != plan.dtype for tensor in (q, k, v)):
        raise ValueError("project_qkv output dtype must match the attention plan")


def validate_projected_q(
    q: torch.Tensor,
    *,
    tokens: int,
    plan: AttentionPlan,
) -> None:
    expected = (tokens, plan.q_heads, plan.head_dim)
    if q.shape != expected:
        raise ValueError(f"project_q returned shape {tuple(q.shape)}, expected {expected}")
    if q.device != plan.device or q.dtype != plan.dtype:
        raise ValueError("project_q must return the planned CUDA dtype/device")


def validate_projected_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tokens: int,
    plan: AttentionPlan,
) -> None:
    expected = (tokens, plan.kv_heads, plan.head_dim)
    if k.shape != expected or v.shape != expected:
        raise ValueError(
            "project_kv returned invalid shapes: "
            f"{tuple(k.shape)}, {tuple(v.shape)}, expected {expected}"
        )
    if any(tensor.device != plan.device for tensor in (k, v)) or any(
        tensor.dtype != plan.dtype for tensor in (k, v)
    ):
        raise ValueError("project_kv must return the planned CUDA dtype/device")


__all__ = [
    "validate_projected_kv",
    "validate_projected_q",
    "validate_projected_qkv",
    "validate_projection_hidden",
]
