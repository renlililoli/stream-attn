from __future__ import annotations

from collections.abc import Iterable

import torch

from ..planner import AttentionPlan


class MaterializedQKVArena:
    """Reusable sequence-sized host Q/K/V storage shared by projected runners."""

    def __init__(
        self,
        *,
        max_q_tokens: int,
        max_kv_tokens: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pin_memory: bool = True,
    ) -> None:
        dimensions = (max_q_tokens, max_kv_tokens, q_heads, kv_heads, head_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("QKV arena dimensions must be positive")
        self.max_q_tokens = max_q_tokens
        self.max_kv_tokens = max_kv_tokens
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.q = torch.empty(
            (max_q_tokens, q_heads, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.k = torch.empty(
            (max_kv_tokens, kv_heads, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.v = torch.empty(
            self.k.shape,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    @classmethod
    def for_plans(
        cls,
        plans: Iterable[AttentionPlan],
        *,
        pin_memory: bool = True,
    ) -> MaterializedQKVArena:
        plans = tuple(plans)
        if not plans:
            raise ValueError("at least one attention plan is required")
        first = plans[0]
        signature = (first.q_heads, first.kv_heads, first.head_dim, first.dtype)
        if any(
            (plan.q_heads, plan.kv_heads, plan.head_dim, plan.dtype) != signature
            for plan in plans[1:]
        ):
            raise ValueError("shared QKV arenas require matching heads, head_dim, and dtype")
        return cls(
            max_q_tokens=max(plan.max_q_tokens for plan in plans),
            max_kv_tokens=max(plan.max_kv_tokens for plan in plans),
            q_heads=first.q_heads,
            kv_heads=first.kv_heads,
            head_dim=first.head_dim,
            dtype=first.dtype,
            pin_memory=pin_memory,
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.q, self.k, self.v))

    def validate_plan(self, plan: AttentionPlan) -> None:
        if (
            plan.q_heads != self.q_heads
            or plan.kv_heads != self.kv_heads
            or plan.head_dim != self.head_dim
            or plan.dtype != self.dtype
        ):
            raise ValueError("attention plan is incompatible with the QKV arena layout")
        if plan.max_q_tokens > self.max_q_tokens or plan.max_kv_tokens > self.max_kv_tokens:
            raise ValueError("attention plan exceeds the QKV arena capacity")

    def views(
        self,
        q_tokens: int,
        kv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 < q_tokens <= self.max_q_tokens:
            raise ValueError("q_tokens must fit the QKV arena")
        if not 0 < kv_tokens <= self.max_kv_tokens:
            raise ValueError("kv_tokens must fit the QKV arena")
        return self.q[:q_tokens], self.k[:kv_tokens], self.v[:kv_tokens]


__all__ = ["MaterializedQKVArena"]
