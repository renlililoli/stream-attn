from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, replace

import torch

from ..projection import ProjectedAttentionRunner
from ..projection.workspace import ProjectionWorkspace
from ..stats import ProjectedAttentionStats
from ..streaming import MultiGpuAttentionPlan
from .types import H3MaterializedProjection


@dataclass(frozen=True)
class QKVProjectionTask:
    start: int
    stop: int
    claim_order: int

    @property
    def tokens(self) -> int:
        return self.stop - self.start


class DynamicQKVProjectionCursor:
    """Thread-safe fixed-size cursor for pointwise QKV projection work."""

    def __init__(self, total_tokens: int, chunk_tokens: int = 4096) -> None:
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.total_tokens = total_tokens
        self.chunk_tokens = chunk_tokens
        self._next_token = 0
        self._claim_order = 0
        self._cancelled = False
        self._lock = threading.Lock()

    def claim(self) -> QKVProjectionTask | None:
        with self._lock:
            if self._cancelled or self._next_token >= self.total_tokens:
                return None
            start = self._next_token
            stop = min(start + self.chunk_tokens, self.total_tokens)
            task = QKVProjectionTask(start, stop, self._claim_order)
            self._next_token = stop
            self._claim_order += 1
            return task

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True


class MultiGpuQKVProjectionRunner:
    """Completion-driven QKV projection into shared pinned host buffers."""

    def __init__(
        self,
        projected_attention: ProjectedAttentionRunner,
        attention_plan: MultiGpuAttentionPlan,
        *,
        hidden_features: int,
        chunk_tokens: int = 4096,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.projected_attention = projected_attention
        self.attention_plan = attention_plan
        self.hidden_features = hidden_features
        self.chunk_tokens = chunk_tokens
        projection_config = replace(
            projected_attention.pipeline_config,
            projection_chunk_tokens=chunk_tokens,
            num_projection_buffers=1,
        )
        self.workspaces: dict[str, ProjectionWorkspace] = {}
        self.estimated_workspace_bytes: dict[str, int] = {}
        element_size = torch.empty((), dtype=attention_plan.dtype).element_size()
        for schedule in attention_plan.schedules:
            device = str(schedule.device)
            with torch.cuda.device(schedule.device):
                self.workspaces[device] = ProjectionWorkspace(
                    hidden_features=hidden_features,
                    dtype=attention_plan.dtype,
                    device=schedule.device,
                    config=projection_config,
                )
            self.estimated_workspace_bytes[device] = chunk_tokens * hidden_features * element_size
        self._executor = ThreadPoolExecutor(
            max_workers=len(attention_plan.schedules),
            thread_name_prefix="seqattn-qkv-projection",
        )
        self._run_lock = threading.Lock()

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _validate_projected_tile(
        self,
        device: torch.device,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tokens: int,
    ) -> None:
        plan = self.attention_plan
        expected_q = (tokens, plan.q_heads, plan.head_dim)
        expected_kv = (tokens, plan.kv_heads, plan.head_dim)
        if q.shape != expected_q:
            raise ValueError(
                f"project_qkv returned q shape {tuple(q.shape)}, expected {expected_q}"
            )
        if k.shape != expected_kv or v.shape != expected_kv:
            raise ValueError(
                "project_qkv returned invalid k/v shapes: "
                f"{tuple(k.shape)}, {tuple(v.shape)}, expected {expected_kv}"
            )
        if any(tensor.device != device for tensor in (q, k, v)):
            raise ValueError(f"project_qkv must return tensors on {device}")
        if any(tensor.dtype != plan.dtype for tensor in (q, k, v)):
            raise ValueError("project_qkv output dtype must match the attention plan")

    def _run_task(
        self,
        device: torch.device,
        hidden_host: torch.Tensor,
        task: QKVProjectionTask,
        projection: H3MaterializedProjection,
        stats: ProjectedAttentionStats,
    ) -> None:
        workspace = self.workspaces[str(device)]
        tile_tokens = task.tokens
        started = time.perf_counter()

        with torch.cuda.stream(workspace.h2d_stream):
            workspace.hidden[0][:tile_tokens].copy_(
                hidden_host[task.start : task.stop],
                non_blocking=hidden_host.is_pinned(),
            )
            workspace.input_ready[0].record(workspace.h2d_stream)

        with torch.cuda.stream(workspace.compute_stream):
            workspace.compute_stream.wait_event(workspace.input_ready[0])
            q, k, v = projection.project_qkv(
                workspace.hidden[0][:tile_tokens],
                task.start,
                task.stop,
            )
            self._validate_projected_tile(device, q, k, v, tile_tokens)
            workspace.projected_ready[0].record(workspace.compute_stream)

        with torch.cuda.stream(workspace.d2h_stream):
            workspace.d2h_stream.wait_event(workspace.projected_ready[0])
            self.projected_attention.q_cpu[task.start : task.stop].copy_(
                q,
                non_blocking=self.projected_attention.q_cpu.is_pinned(),
            )
            self.projected_attention.k_cpu[task.start : task.stop].copy_(
                k,
                non_blocking=self.projected_attention.k_cpu.is_pinned(),
            )
            self.projected_attention.v_cpu[task.start : task.stop].copy_(
                v,
                non_blocking=self.projected_attention.v_cpu.is_pinned(),
            )
            workspace.copy_done[0].record(workspace.d2h_stream)

        workspace.keepalive[0] = (q, k, v)
        workspace.copy_done[0].synchronize()
        workspace.keepalive[0] = None
        stats.backend = "triton"
        stats.projection_seconds += time.perf_counter() - started
        stats.projection_chunks += 1
        stats.projection_tokens += tile_tokens
        stats.projection_hidden_h2d_bytes += (
            tile_tokens * self.hidden_features * hidden_host.element_size()
        )
        stats.projection_qkv_d2h_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in (q, k, v)
        )

    def run(
        self,
        hidden_host: torch.Tensor,
        projections_by_device: Mapping[str, H3MaterializedProjection],
        *,
        stats: ProjectedAttentionStats,
        per_device_stats: Mapping[str, ProjectedAttentionStats],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("MultiGpuQKVProjectionRunner is single-flight")
        try:
            self.projected_attention._validate_hidden(hidden_host)
            tokens = hidden_host.shape[0]
            if tokens != self.attention_plan.max_q_tokens:
                raise ValueError("hidden token count must match the multi-GPU attention plan")
            expected = {str(device) for device in self.attention_plan.devices}
            if set(projections_by_device) != expected or set(per_device_stats) != expected:
                raise ValueError("projections and stats must contain every planned device")

            cursor = DynamicQKVProjectionCursor(tokens, self.chunk_tokens)
            start_barrier = threading.Barrier(len(self.attention_plan.schedules))
            failure_lock = threading.Lock()
            failures: list[Exception] = []

            def run_device(index: int) -> None:
                schedule = self.attention_plan.schedules[index]
                device = schedule.device
                device_name = str(device)
                start_barrier.wait()
                try:
                    with (
                        torch.inference_mode(),
                        torch.cuda.device(device),
                        projections_by_device[device_name].context(),
                    ):
                        while True:
                            task = cursor.claim()
                            if task is None:
                                break
                            self._run_task(
                                device,
                                hidden_host,
                                task,
                                projections_by_device[device_name],
                                per_device_stats[device_name],
                            )
                except Exception as error:  # noqa: BLE001 - propagate the original worker error.
                    cursor.cancel()
                    with suppress(Exception):
                        torch.cuda.synchronize(device)
                    with failure_lock:
                        if not failures:
                            failures.append(error)

            started = time.perf_counter()
            futures = [
                self._executor.submit(run_device, index)
                for index in range(len(self.attention_plan.schedules))
            ]
            for future in futures:
                future.result()
            elapsed = time.perf_counter() - started
            if failures:
                raise failures[0]

            q_cpu = self.projected_attention.q_cpu[:tokens]
            k_cpu = self.projected_attention.k_cpu[:tokens]
            v_cpu = self.projected_attention.v_cpu[:tokens]
            stats.backend = "triton"
            stats.projection_seconds += elapsed
            stats.projection_chunks += (tokens + self.chunk_tokens - 1) // self.chunk_tokens
            stats.projection_tokens += tokens
            stats.projection_hidden_h2d_bytes += (
                tokens * self.hidden_features * hidden_host.element_size()
            )
            stats.projection_qkv_d2h_bytes += sum(
                tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
            )
            stats.qkv_host_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
            )
            return q_cpu, k_cpu, v_cpu
        finally:
            self._run_lock.release()


__all__ = [
    "DynamicQKVProjectionCursor",
    "MultiGpuQKVProjectionRunner",
    "QKVProjectionTask",
]
