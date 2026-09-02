from __future__ import annotations

from dataclasses import dataclass

import torch

from ..plan import AttentionPlan


@dataclass(frozen=True)
class QueryTask:
    """One independently executable query range and its complete KV segment."""

    q_start: int
    q_stop: int
    k_start: int
    k_stop: int
    q_local_offset: int
    causal_shift: int
    segment_id: int = -1
    device_id: int = -1
    claim_order: int = -1
    requested_q: int | None = None
    segment_clamped: bool = False
    tail_clamped: bool = False

    @property
    def q_tokens(self) -> int:
        return self.q_stop - self.q_start

    @property
    def k_tokens(self) -> int:
        return self.k_stop - self.k_start

    def validate(self) -> None:
        if self.q_start < 0 or self.q_stop <= self.q_start:
            raise ValueError("query task must contain a non-empty non-negative Q range")
        if self.k_start < 0 or self.k_stop <= self.k_start:
            raise ValueError("query task must contain a non-empty non-negative K/V range")
        if self.q_local_offset < 0:
            raise ValueError("query task q_local_offset must be non-negative")
        if self.requested_q is not None and self.requested_q < self.q_tokens:
            raise ValueError("query task requested_q cannot be smaller than its Q range")


def build_query_tasks(
    q_bounds: list[int],
    k_bounds: list[int],
    *,
    q_chunk_tokens: int,
    range_start: int | None = None,
    range_stop: int | None = None,
) -> tuple[QueryTask, ...]:
    """Partition a contiguous global Q range without crossing packed segments."""

    if q_chunk_tokens <= 0:
        raise ValueError("q_chunk_tokens must be positive")
    if len(q_bounds) != len(k_bounds) or len(q_bounds) < 2:
        raise ValueError("q_bounds and k_bounds must describe the same non-empty batch")
    total_q = q_bounds[-1]
    range_start = 0 if range_start is None else range_start
    range_stop = total_q if range_stop is None else range_stop
    if not 0 <= range_start <= range_stop <= total_q:
        raise ValueError(f"query range must lie within [0, {total_q}]")

    tasks: list[QueryTask] = []
    for segment_id, (q_start, q_stop, k_start, k_stop) in enumerate(
        zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:])
    ):
        tile_range_start = max(q_start, range_start)
        tile_range_stop = min(q_stop, range_stop)
        if tile_range_start >= tile_range_stop:
            continue
        q_length = q_stop - q_start
        k_length = k_stop - k_start
        if k_length <= 0:
            raise ValueError("a non-empty query segment requires a non-empty K/V segment")
        causal_shift = k_length - q_length
        for tile_start in range(tile_range_start, tile_range_stop, q_chunk_tokens):
            tile_stop = min(tile_start + q_chunk_tokens, tile_range_stop)
            tasks.append(
                QueryTask(
                    q_start=tile_start,
                    q_stop=tile_stop,
                    k_start=k_start,
                    k_stop=k_stop,
                    q_local_offset=tile_start - q_start,
                    causal_shift=causal_shift,
                    segment_id=segment_id,
                )
            )
    return tuple(tasks)


def validate_query_task_inputs(
    plan: AttentionPlan,
    q_cpu: torch.Tensor,
    k_cpu: torch.Tensor,
    v_cpu: torch.Tensor,
    query_tasks: tuple[QueryTask, ...],
) -> None:
    if any(tensor.device.type != "cpu" for tensor in (q_cpu, k_cpu, v_cpu)):
        raise ValueError("scheduled query task inputs must be CPU-backed")
    if q_cpu.shape[1:] != (plan.q_heads, plan.head_dim):
        raise ValueError("q shape does not match the runner plan")
    if k_cpu.shape != v_cpu.shape or k_cpu.shape[1:] != (plan.kv_heads, plan.head_dim):
        raise ValueError("k/v shape does not match the runner plan")
    if any(tensor.dtype != plan.dtype for tensor in (q_cpu, k_cpu, v_cpu)):
        raise ValueError("input dtype does not match the runner plan")
    if q_cpu.shape[0] > plan.max_q_tokens or k_cpu.shape[0] > plan.max_kv_tokens:
        raise ValueError("input token count exceeds the runner plan")

    previous_stop = 0
    for task in query_tasks:
        task.validate()
        if task.q_start < previous_stop:
            raise ValueError("query tasks must be ordered and non-overlapping")
        if task.q_stop > q_cpu.shape[0] or task.k_stop > k_cpu.shape[0]:
            raise ValueError("query task exceeds the provided Q/K/V tensors")
        if task.q_tokens > plan.q_chunk_tokens:
            raise ValueError("query task exceeds the runner q_chunk_tokens")
        segment_q_tokens = task.k_tokens - task.causal_shift
        if task.q_local_offset + task.q_tokens > segment_q_tokens:
            raise ValueError("query task exceeds its packed query segment")
        previous_stop = task.q_stop


__all__ = ["QueryTask", "build_query_tasks", "validate_query_task_inputs"]
