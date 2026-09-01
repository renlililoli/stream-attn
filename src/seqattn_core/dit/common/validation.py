from __future__ import annotations

import torch

from ...planner import AttentionPlan


def validate_hidden_host(
    hidden_host: torch.Tensor,
    *,
    plan: AttentionPlan,
    hidden_features: int,
    require_pinned: bool,
    name: str = "hidden_host",
) -> None:
    if hidden_host.device.type != "cpu" or hidden_host.ndim != 2:
        raise ValueError(f"{name} must use CPU [tokens, hidden_features] layout")
    if hidden_host.shape[1] != hidden_features:
        raise ValueError(f"{name} feature size does not match the runner")
    if hidden_host.shape[0] > plan.max_q_tokens or hidden_host.shape[0] > plan.max_kv_tokens:
        raise ValueError(f"{name} token count exceeds the runner plan")
    if hidden_host.dtype != plan.dtype:
        raise ValueError(f"{name} dtype does not match the runner plan")
    if not hidden_host.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if require_pinned and not hidden_host.is_pinned():
        raise ValueError(f"asynchronous execution requires pinned {name}")


def require_distinct_storage(source: torch.Tensor, destination: torch.Tensor) -> None:
    if source.shape != destination.shape:
        raise ValueError("source and destination hidden tensors must have identical shapes")
    if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        raise ValueError("recompute requires distinct source and destination storage")


__all__ = ["require_distinct_storage", "validate_hidden_host"]
