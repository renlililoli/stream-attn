from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass

import torch

from .types import DeviceTileOp


@dataclass
class TiledStageStats:
    wall_seconds: float = 0.0
    chunks: int = 0
    tokens: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0


class TiledStageWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        chunk_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        num_buffers: int = 2,
    ) -> None:
        if hidden_features <= 0 or chunk_tokens <= 0:
            raise ValueError("hidden_features and chunk_tokens must be positive")
        if num_buffers not in {1, 2}:
            raise ValueError("num_buffers must be 1 or 2")
        self.hidden_features = hidden_features
        self.chunk_tokens = chunk_tokens
        self.dtype = dtype
        self.device = device
        self.input = [
            torch.empty((chunk_tokens, hidden_features), dtype=dtype, device=device)
            for _ in range(num_buffers)
        ]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.input_ready = [torch.cuda.Event() for _ in self.input]
        self.output_ready = [torch.cuda.Event() for _ in self.input]
        self.copy_done = [torch.cuda.Event() for _ in self.input]
        self.busy = [False for _ in self.input]
        self.keepalive: list[torch.Tensor | None] = [None for _ in self.input]


class TiledHostStageRunner:
    """Run a pointwise model stage over bounded CUDA tiles."""

    def __init__(
        self,
        *,
        hidden_features: int,
        chunk_tokens: int,
        dtype: torch.dtype,
        device: torch.device | str,
        num_buffers: int = 2,
        require_pinned_hidden: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("tiled host stages require a CUDA device")
        self.require_pinned_hidden = require_pinned_hidden
        self.workspace = TiledStageWorkspace(
            hidden_features=hidden_features,
            chunk_tokens=chunk_tokens,
            dtype=dtype,
            device=self.device,
            num_buffers=num_buffers,
        )

    def _validate(self, tensor: torch.Tensor, *, name: str) -> None:
        workspace = self.workspace
        if tensor.device.type != "cpu" or tensor.ndim != 2:
            raise ValueError(f"{name} must use CPU [tokens, hidden_features] layout")
        if tensor.shape[1] != workspace.hidden_features or tensor.dtype != workspace.dtype:
            raise ValueError(f"{name} shape/dtype does not match the tiled stage")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if self.require_pinned_hidden and not tensor.is_pinned():
            raise ValueError(f"asynchronous tiled execution requires pinned {name}")

    def _run_range(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        operation: DeviceTileOp,
        *,
        range_start: int,
        range_stop: int,
        stats: TiledStageStats,
    ) -> None:
        workspace = self.workspace
        chunk = workspace.chunk_tokens
        for chunk_index, start in enumerate(range(range_start, range_stop, chunk)):
            stop = min(start + chunk, range_stop)
            tokens = stop - start
            slot = chunk_index % len(workspace.input)
            if workspace.busy[slot]:
                workspace.copy_done[slot].synchronize()
                workspace.keepalive[slot] = None
            with torch.cuda.stream(workspace.h2d_stream):
                workspace.input[slot][:tokens].copy_(
                    source_hidden_host[start:stop],
                    non_blocking=source_hidden_host.is_pinned(),
                )
                workspace.input_ready[slot].record(workspace.h2d_stream)
            with torch.cuda.stream(workspace.compute_stream):
                workspace.compute_stream.wait_event(workspace.input_ready[slot])
                output = operation(workspace.input[slot][:tokens], start, stop)
                expected = (tokens, workspace.hidden_features)
                if output.shape != expected:
                    raise ValueError(
                        f"tiled stage returned shape {tuple(output.shape)}, expected {expected}"
                    )
                if output.device != workspace.device or output.dtype != workspace.dtype:
                    raise ValueError("tiled stage must return the planned CUDA dtype/device")
                workspace.output_ready[slot].record(workspace.compute_stream)
            with torch.cuda.stream(workspace.d2h_stream):
                workspace.d2h_stream.wait_event(workspace.output_ready[slot])
                destination_hidden_host[start:stop].copy_(
                    output,
                    non_blocking=destination_hidden_host.is_pinned(),
                )
                workspace.copy_done[slot].record(workspace.d2h_stream)
            workspace.keepalive[slot] = output
            workspace.busy[slot] = True
            stats.chunks += 1
            stats.tokens += tokens
            stats.h2d_bytes += (
                source_hidden_host[start:stop].numel() * source_hidden_host.element_size()
            )
            stats.d2h_bytes += output.numel() * output.element_size()
        workspace.d2h_stream.synchronize()
        workspace.keepalive[:] = [None] * len(workspace.keepalive)
        workspace.busy[:] = [False] * len(workspace.busy)

    def _recover_after_failure(self) -> None:
        with suppress(Exception):
            torch.cuda.synchronize(self.device)
        self.workspace.keepalive[:] = [None] * len(self.workspace.keepalive)
        self.workspace.busy[:] = [False] * len(self.workspace.busy)

    @torch.inference_mode()
    def run(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        operation: DeviceTileOp,
        *,
        stats: TiledStageStats | None = None,
    ) -> torch.Tensor:
        self._validate(source_hidden_host, name="source_hidden_host")
        self._validate(destination_hidden_host, name="destination_hidden_host")
        if source_hidden_host.shape != destination_hidden_host.shape:
            raise ValueError("source and destination hidden tensors must have identical shapes")
        stats = TiledStageStats() if stats is None else stats
        started = time.perf_counter()
        try:
            self._run_range(
                source_hidden_host,
                destination_hidden_host,
                operation,
                range_start=0,
                range_stop=source_hidden_host.shape[0],
                stats=stats,
            )
        except Exception:
            self._recover_after_failure()
            raise
        stats.wall_seconds += time.perf_counter() - started
        return destination_hidden_host
__all__ = [
    "TiledHostStageRunner",
    "TiledStageStats",
]
