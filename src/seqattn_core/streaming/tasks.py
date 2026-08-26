from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTask:
    """One independently executable query range and its complete KV segment."""

    q_start: int
    q_stop: int
    k_start: int
    k_stop: int
    q_local_offset: int
    causal_shift: int

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
    for q_start, q_stop, k_start, k_stop in zip(
        q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
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
                )
            )
    return tuple(tasks)


__all__ = ["QueryTask", "build_query_tasks"]
