from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress

import torch

from ..config import ProjectionPipelineConfig
from ..plan import AttentionPlan


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
            (max_q_tokens, q_heads, head_dim), dtype=dtype, device="cpu", pin_memory=pin_memory
        )
        self.k = torch.empty(
            (max_kv_tokens, kv_heads, head_dim), dtype=dtype, device="cpu", pin_memory=pin_memory
        )
        self.v = torch.empty(self.k.shape, dtype=dtype, device="cpu", pin_memory=pin_memory)

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


class ProjectionWorkspace:
    """Persistent streams and staging slots for one materialized hidden source."""

    def __init__(
        self,
        *,
        hidden_features: int,
        dtype: torch.dtype,
        device: torch.device,
        config: ProjectionPipelineConfig,
    ) -> None:
        self.hidden_features = hidden_features
        self.hidden = [
            torch.empty(
                (config.projection_tile_tokens, hidden_features), dtype=dtype, device=device
            )
            for _ in range(config.num_projection_buffers)
        ]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.input_ready = [torch.cuda.Event() for _ in self.hidden]
        self.projected_ready = [torch.cuda.Event() for _ in self.hidden]
        self.copy_done = [torch.cuda.Event() for _ in self.hidden]
        self.busy = [False for _ in self.hidden]
        self.keepalive: list[object | None] = [None for _ in self.hidden]

    def release_slot(self, slot: int) -> None:
        self.keepalive[slot] = None
        self.busy[slot] = False

    def reset_slots(self) -> None:
        for slot in range(len(self.hidden)):
            self.release_slot(slot)

    def recover(self) -> None:
        for stream in (self.h2d_stream, self.compute_stream, self.d2h_stream):
            with suppress(Exception):
                stream.synchronize()
        self.reset_slots()


class RecomputeWorkspace:
    """Persistent hidden staging used by direct-write Q/KV recomputation."""

    def __init__(
        self,
        *,
        hidden_features: int,
        staging_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.hidden_features = hidden_features
        self.staging_tokens = staging_tokens
        self.hidden = torch.empty((staging_tokens, hidden_features), dtype=dtype, device=device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.hidden_ready = torch.cuda.Event()
        self.hidden_free = torch.cuda.Event()
        self.hidden_has_pending_compute = False

    def recover(self) -> None:
        with suppress(Exception):
            self.h2d_stream.synchronize()
        self.hidden_has_pending_compute = False


class CrossRecomputeWorkspace:
    """Independent query and context staging for recomputed cross-attention."""

    def __init__(
        self,
        *,
        query_hidden_features: int,
        context_hidden_features: int,
        q_staging_tokens: int,
        kv_staging_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.query = RecomputeWorkspace(
            hidden_features=query_hidden_features,
            staging_tokens=q_staging_tokens,
            dtype=dtype,
            device=device,
        )
        self.context = RecomputeWorkspace(
            hidden_features=context_hidden_features,
            staging_tokens=kv_staging_tokens,
            dtype=dtype,
            device=device,
        )

    def recover(self) -> None:
        self.query.recover()
        self.context.recover()


__all__ = [
    "CrossRecomputeWorkspace",
    "MaterializedQKVArena",
    "ProjectionWorkspace",
    "RecomputeWorkspace",
]
