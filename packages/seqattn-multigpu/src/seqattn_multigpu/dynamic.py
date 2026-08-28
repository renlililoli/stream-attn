from __future__ import annotations

import math
import threading
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import pairwise

from seqattn_core._plugin_api import QueryTask, QueryTaskMeasurement


def _align_up(value: float, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _align_down(value: float, alignment: int) -> int:
    return math.floor(value / alignment) * alignment


@dataclass(frozen=True)
class DynamicScheduleConfig:
    ema_alpha: float = 0.20
    safety_factor: float = 1.15
    change_threshold: float = 0.125
    max_step_ratio: float = 1.25
    tail_balance_factor: float = 1.25
    enable_task_trace: bool = False

    def validate(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be within (0, 1]")
        if self.safety_factor <= 0:
            raise ValueError("safety_factor must be positive")
        if not 0.0 <= self.change_threshold < 1.0:
            raise ValueError("change_threshold must be within [0, 1)")
        if self.max_step_ratio <= 1.0:
            raise ValueError("max_step_ratio must be greater than 1")
        if self.tail_balance_factor <= 0:
            raise ValueError("tail_balance_factor must be positive")


@dataclass(frozen=True)
class DynamicWorkloadSignature:
    q_segment_lengths: tuple[int, ...]
    k_segment_lengths: tuple[int, ...]
    q_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    kernel_profile: tuple[int, int, int, int]
    consumer_mode: str


@dataclass(frozen=True)
class DynamicControllerSnapshot:
    effective_tflops_ema: float
    h2d_gbps_ema: float
    d2h_gbps_ema: float
    task_elapsed_ema: float
    q_current: int


class DynamicQController:
    """Per-device dynamic Q block controller retained across repeated calls."""

    def __init__(
        self,
        *,
        initial_q_tokens: int,
        q_min_tokens: int,
        q_capacity_tokens: int,
        block_m: int,
        compute_tflops: float,
        h2d_gbps: float,
        d2h_gbps: float,
        q_heads: int,
        kv_heads: int,
        element_size: int,
        config: DynamicScheduleConfig,
    ) -> None:
        config.validate()
        if block_m <= 0:
            raise ValueError("block_m must be positive")
        self.config = config
        self.block_m = block_m
        self.alignment = min(block_m, q_capacity_tokens)
        self.q_min_tokens = _align_up(q_min_tokens, self.alignment)
        self.q_capacity_tokens = _align_down(q_capacity_tokens, self.alignment)
        if self.q_capacity_tokens <= 0 or self.q_min_tokens > self.q_capacity_tokens:
            raise ValueError("dynamic Q bounds do not contain one aligned block")
        if not 0 < initial_q_tokens <= q_capacity_tokens:
            raise ValueError("initial_q_tokens must lie within Q capacity")
        self.initial_q_tokens = min(
            self.q_capacity_tokens,
            max(self.q_min_tokens, _align_up(initial_q_tokens, self.alignment)),
        )
        if compute_tflops <= 0 or h2d_gbps <= 0 or d2h_gbps <= 0:
            raise ValueError("initial performance estimates must be positive")
        self.initial_compute_tflops = compute_tflops
        self.initial_h2d_gbps = h2d_gbps
        self.initial_d2h_gbps = d2h_gbps
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.element_size = element_size
        self.signature: Hashable | None = None
        self.reset()

    def reset(self) -> None:
        self.effective_tflops_ema = self.initial_compute_tflops
        self.h2d_gbps_ema = self.initial_h2d_gbps
        self.d2h_gbps_ema = self.initial_d2h_gbps
        self.task_elapsed_ema = 0.0
        self.q_current = self.initial_q_tokens

    def reset_for_signature(self, signature: Hashable) -> None:
        if signature != self.signature:
            self.signature = signature
            self.reset()

    @staticmethod
    def _ema(previous: float, sample: float, alpha: float) -> float:
        return sample if previous <= 0 else previous + alpha * (sample - previous)

    def _target_q_tokens(self) -> int:
        q_knee = (
            self.config.safety_factor
            * self.effective_tflops_ema
            * 1e12
            * self.kv_heads
            * self.element_size
            / (2 * self.q_heads * self.h2d_gbps_ema * 1e9)
        )
        target = _align_up(max(q_knee, self.q_min_tokens), self.alignment)
        return min(self.q_capacity_tokens, max(self.q_min_tokens, target))

    def observe(
        self,
        measurement: QueryTaskMeasurement,
        *,
        update_compute: bool,
    ) -> tuple[int, int]:
        alpha = self.config.ema_alpha
        if measurement.h2d_seconds > 0 and measurement.h2d_bytes > 0:
            sample = measurement.h2d_bytes / measurement.h2d_seconds / 1e9
            self.h2d_gbps_ema = self._ema(self.h2d_gbps_ema, sample, alpha)
        if measurement.d2h_seconds > 0 and measurement.d2h_bytes > 0:
            sample = measurement.d2h_bytes / measurement.d2h_seconds / 1e9
            self.d2h_gbps_ema = self._ema(self.d2h_gbps_ema, sample, alpha)
        if measurement.elapsed_seconds > 0:
            self.task_elapsed_ema = self._ema(
                self.task_elapsed_ema,
                measurement.elapsed_seconds,
                alpha,
            )
        if update_compute and measurement.attention_seconds > 0 and measurement.attention_flops > 0:
            sample = measurement.attention_flops / measurement.attention_seconds / 1e12
            self.effective_tflops_ema = self._ema(
                self.effective_tflops_ema,
                sample,
                alpha,
            )

        previous = self.q_current
        target = self._target_q_tokens()
        if abs(target - previous) / previous < self.config.change_threshold:
            return previous, previous

        if target > previous:
            step_limit = _align_up(previous * self.config.max_step_ratio, self.alignment)
            updated = min(target, step_limit, self.q_capacity_tokens)
        else:
            step_limit = _align_down(previous / self.config.max_step_ratio, self.alignment)
            updated = max(target, step_limit, self.q_min_tokens)
        self.q_current = updated
        return previous, updated

    def snapshot(self) -> DynamicControllerSnapshot:
        return DynamicControllerSnapshot(
            effective_tflops_ema=self.effective_tflops_ema,
            h2d_gbps_ema=self.h2d_gbps_ema,
            d2h_gbps_ema=self.d2h_gbps_ema,
            task_elapsed_ema=self.task_elapsed_ema,
            q_current=self.q_current,
        )


class DynamicQueryCursor:
    """Thread-safe segment-aware dispatcher for one attention invocation."""

    def __init__(
        self,
        q_bounds: tuple[int, ...] | list[int],
        k_bounds: tuple[int, ...] | list[int],
        *,
        active_workers: int,
        tail_balance_factor: float = 1.25,
    ) -> None:
        if len(q_bounds) != len(k_bounds) or len(q_bounds) < 2:
            raise ValueError("q_bounds and k_bounds must describe the same non-empty batch")
        if any(stop < start for start, stop in pairwise(q_bounds)):
            raise ValueError("q_bounds must be non-decreasing")
        if any(stop < start for start, stop in pairwise(k_bounds)):
            raise ValueError("k_bounds must be non-decreasing")
        if active_workers <= 0:
            raise ValueError("active_workers must be positive")
        if tail_balance_factor <= 0:
            raise ValueError("tail_balance_factor must be positive")
        self.q_bounds = tuple(q_bounds)
        self.k_bounds = tuple(k_bounds)
        self.tail_balance_factor = tail_balance_factor
        self._segment_id = 0
        self._q_offset = self.q_bounds[0]
        self._remaining_unclaimed = self.q_bounds[-1] - self.q_bounds[0]
        self._active_workers = active_workers
        self._retired_workers: set[int] = set()
        self._claim_order = 0
        self._cancelled = False
        self._lock = threading.Lock()

    @property
    def remaining_unclaimed(self) -> int:
        with self._lock:
            return self._remaining_unclaimed

    @property
    def active_workers(self) -> int:
        with self._lock:
            return self._active_workers

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True

    def retire(self, device_id: int) -> None:
        with self._lock:
            if device_id not in self._retired_workers:
                self._retired_workers.add(device_id)
                self._active_workers = max(0, self._active_workers - 1)

    def _advance_empty_segments(self) -> None:
        last_segment = len(self.q_bounds) - 1
        while self._segment_id < last_segment:
            segment_stop = self.q_bounds[self._segment_id + 1]
            if self._q_offset < segment_stop:
                break
            self._segment_id += 1
            if self._segment_id < last_segment:
                self._q_offset = self.q_bounds[self._segment_id]

    def claim(self, device_id: int, requested_q: int) -> QueryTask | None:
        if requested_q <= 0:
            raise ValueError("requested_q must be positive")
        with self._lock:
            if self._cancelled or self._remaining_unclaimed <= 0:
                return None
            self._advance_empty_segments()
            if self._segment_id >= len(self.q_bounds) - 1:
                return None

            q_segment_start = self.q_bounds[self._segment_id]
            q_segment_stop = self.q_bounds[self._segment_id + 1]
            k_start = self.k_bounds[self._segment_id]
            k_stop = self.k_bounds[self._segment_id + 1]
            if k_stop <= k_start:
                raise ValueError("a non-empty query segment requires a non-empty K/V segment")
            segment_remaining = q_segment_stop - self._q_offset
            active_workers = max(self._active_workers, 1)
            tail_workers = min(
                active_workers,
                max(1, math.ceil(self._remaining_unclaimed / requested_q)),
            )
            tail_limit = math.ceil(
                self._remaining_unclaimed / tail_workers * self.tail_balance_factor
            )
            actual_q = min(requested_q, segment_remaining, max(1, tail_limit))
            q_start = self._q_offset
            q_stop = q_start + actual_q
            segment_clamped = actual_q < requested_q and actual_q == segment_remaining
            tail_clamped = actual_q < requested_q and actual_q == tail_limit
            task = QueryTask(
                q_start=q_start,
                q_stop=q_stop,
                k_start=k_start,
                k_stop=k_stop,
                q_local_offset=q_start - q_segment_start,
                causal_shift=(k_stop - k_start) - (q_segment_stop - q_segment_start),
                segment_id=self._segment_id,
                device_id=device_id,
                claim_order=self._claim_order,
                requested_q=requested_q,
                segment_clamped=segment_clamped,
                tail_clamped=tail_clamped,
            )
            self._claim_order += 1
            self._q_offset = q_stop
            self._remaining_unclaimed -= actual_q
            return task


__all__ = [
    "DynamicControllerSnapshot",
    "DynamicQController",
    "DynamicQueryCursor",
    "DynamicScheduleConfig",
    "DynamicWorkloadSignature",
]
