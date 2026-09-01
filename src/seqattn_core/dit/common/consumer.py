from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .types import AttentionEpilogue

if TYPE_CHECKING:
    from ...streaming.tasks import QueryTask


class AttentionOutputWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        output_chunk_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        num_output_buffers: int = 2,
    ) -> None:
        if hidden_features <= 0 or output_chunk_tokens <= 0:
            raise ValueError("hidden_features and output_chunk_tokens must be positive")
        if num_output_buffers not in {1, 2}:
            raise ValueError("num_output_buffers must be 1 or 2")
        self.hidden_features = hidden_features
        self.output_chunk_tokens = output_chunk_tokens
        self.dtype = dtype
        self.device = device
        self.output = [
            torch.empty(
                (output_chunk_tokens, hidden_features),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_output_buffers)
        ]
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.output_ready = [torch.cuda.Event() for _ in self.output]
        self.output_free = [torch.cuda.Event() for _ in self.output]
        self.output_pending = [False for _ in self.output]
        self.task_d2h_start = torch.cuda.Event(enable_timing=True)
        self.task_done = torch.cuda.Event(enable_timing=True)


class AttentionOutputConsumer:
    """Apply a model epilogue on device and write bounded tiles to pinned host."""

    def __init__(self, workspace: AttentionOutputWorkspace) -> None:
        self.workspace = workspace
        self.destination_hidden_host: torch.Tensor | None = None
        self.residual_hidden_host: torch.Tensor | None = None
        self.epilogue: AttentionEpilogue | None = None
        self.total_tokens = 0
        self.next_token = 0
        self.output_index = 0
        self.d2h_bytes = 0
        self._task_active = False
        self._task_d2h_started = False
        self._task_q_tokens = 0

    def reset(
        self,
        *,
        destination_hidden_host: torch.Tensor,
        residual_hidden_host: torch.Tensor,
        epilogue: AttentionEpilogue,
        range_start: int = 0,
        range_stop: int | None = None,
    ) -> None:
        if (
            destination_hidden_host.device.type != "cpu"
            or residual_hidden_host.device.type != "cpu"
            or destination_hidden_host.ndim != 2
            or residual_hidden_host.ndim != 2
        ):
            raise ValueError(
                "attention consumer hidden tensors must use CPU [tokens, hidden_features] layout"
            )
        if destination_hidden_host.shape != residual_hidden_host.shape:
            raise ValueError("destination and residual hidden tensors must have identical shapes")
        if destination_hidden_host.shape[1] != self.workspace.hidden_features:
            raise ValueError("hidden feature size does not match the output consumer")
        if (
            destination_hidden_host.dtype != self.workspace.dtype
            or residual_hidden_host.dtype != self.workspace.dtype
        ):
            raise ValueError("hidden dtype does not match the output consumer")
        if not destination_hidden_host.is_contiguous() or not residual_hidden_host.is_contiguous():
            raise ValueError("attention consumer hidden tensors must be contiguous")
        range_stop = destination_hidden_host.shape[0] if range_stop is None else range_stop
        if not 0 <= range_start < range_stop <= destination_hidden_host.shape[0]:
            raise ValueError("consumer range must be a non-empty hidden tensor slice")
        self.destination_hidden_host = destination_hidden_host
        self.residual_hidden_host = residual_hidden_host
        self.epilogue = epilogue
        self.total_tokens = range_stop
        self.next_token = range_start
        self.output_index = 0
        self.d2h_bytes = 0
        self._task_active = False
        self._task_d2h_started = False
        self._task_q_tokens = 0

    def begin_task(self, task: QueryTask) -> None:
        if self._task_active:
            raise RuntimeError("cannot begin a task before finishing the current task")
        self.total_tokens = task.q_stop
        self.next_token = task.q_start
        self._task_active = True
        self._task_d2h_started = False
        self._task_q_tokens = task.q_tokens

    def __call__(self, attention: torch.Tensor, start: int, stop: int) -> None:
        if self.destination_hidden_host is None or self.residual_hidden_host is None:
            raise RuntimeError("attention consumer must be reset before use")
        if self.epilogue is None:
            raise RuntimeError("attention consumer epilogue is not configured")
        if start != self.next_token or stop <= start:
            raise ValueError(
                f"attention output ranges must be contiguous, got [{start}, {stop}) "
                f"after token {self.next_token}"
            )
        tokens = stop - start
        if tokens > self.workspace.output_chunk_tokens:
            raise ValueError("attention output exceeds the preallocated output chunk")
        compute_stream = torch.cuda.current_stream(self.workspace.device)
        slot_index = self.output_index % len(self.workspace.output)
        if self.workspace.output_pending[slot_index]:
            compute_stream.wait_event(self.workspace.output_free[slot_index])
        result = self.epilogue(attention, self.residual_hidden_host, start, stop)
        expected = (tokens, self.workspace.hidden_features)
        if result.shape != expected:
            raise ValueError(
                f"attention epilogue returned {tuple(result.shape)}, expected {expected}"
            )
        if result.device != self.workspace.device or result.dtype != self.workspace.dtype:
            raise ValueError("attention epilogue must return the planned CUDA dtype/device")
        output = self.workspace.output[slot_index][:tokens]
        output.copy_(result)
        self.workspace.output_ready[slot_index].record(compute_stream)
        with torch.cuda.stream(self.workspace.d2h_stream):
            if self._task_active and not self._task_d2h_started:
                self.workspace.task_d2h_start.record(self.workspace.d2h_stream)
                self._task_d2h_started = True
            self.workspace.d2h_stream.wait_event(self.workspace.output_ready[slot_index])
            self.destination_hidden_host[start:stop].copy_(
                output,
                non_blocking=self.destination_hidden_host.is_pinned(),
            )
            self.workspace.output_free[slot_index].record(self.workspace.d2h_stream)
        self.workspace.output_pending[slot_index] = True
        self.output_index += 1
        self.next_token = stop
        self.d2h_bytes += output.numel() * output.element_size()

    def finish(self) -> None:
        if self.next_token != self.total_tokens:
            raise ValueError(
                f"attention consumer received {self.next_token} of {self.total_tokens} tokens"
            )

    def finish_task(self) -> torch.cuda.Event:
        if not self._task_active:
            raise RuntimeError("cannot finish a task before begin_task()")
        self.finish()
        with torch.cuda.stream(self.workspace.d2h_stream):
            self.workspace.task_done.record(self.workspace.d2h_stream)
        self._task_active = False
        return self.workspace.task_done

    def task_d2h_seconds(self) -> float:
        if not self._task_d2h_started:
            return 0.0
        return self.workspace.task_d2h_start.elapsed_time(self.workspace.task_done) / 1000.0

    def task_d2h_bytes(self) -> int:
        return (
            self._task_q_tokens
            * self.workspace.hidden_features
            * self.workspace.output[0].element_size()
        )

    def synchronize(self) -> None:
        self.workspace.d2h_stream.synchronize()


__all__ = ["AttentionOutputConsumer", "AttentionOutputWorkspace"]
