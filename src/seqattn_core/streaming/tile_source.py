from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Protocol

import torch

from ..stats import RecomputedAttentionStats, StreamingAttentionStats
from .workspace import CudaWorkspace

if TYPE_CHECKING:
    from ..projection.recompute_workspace import RecomputeWorkspace
    from ..projection.types import KVTileProjector, QTileProjector


class QKVTileSource(Protocol):
    def load_q(
        self,
        destination: torch.Tensor,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None: ...

    def load_kv(
        self,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        buffer_index: int,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None: ...

    def release_q(self, compute_stream: torch.cuda.Stream) -> None: ...

    def release_kv(self, buffer_index: int, compute_stream: torch.cuda.Stream) -> None: ...


class HostQKVTileSource:
    def __init__(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        workspace: CudaWorkspace,
        *,
        enable_nvtx: bool,
    ) -> None:
        self.q_cpu = q_cpu
        self.k_cpu = k_cpu
        self.v_cpu = v_cpu
        self.workspace = workspace
        self.enable_nvtx = enable_nvtx

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.enable_nvtx else nullcontext()

    def load_q(
        self,
        destination: torch.Tensor,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None:
        workspace = self.workspace
        tokens = stop - start
        with self._range("seqattn:q_h2d"), torch.cuda.stream(workspace.h2d_stream):
            if workspace.q_has_pending_compute:
                workspace.h2d_stream.wait_event(workspace.q_free)
            destination.copy_(
                self.q_cpu[start:stop],
                non_blocking=self.q_cpu.is_pinned(),
            )
            workspace.q_ready.record(workspace.h2d_stream)
        compute_stream.wait_event(workspace.q_ready)
        stats.h2d_bytes += (
            tokens * self.q_cpu.shape[1] * self.q_cpu.shape[2] * self.q_cpu.element_size()
        )

    def load_kv(
        self,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        buffer_index: int,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None:
        workspace = self.workspace
        tokens = stop - start
        with self._range("seqattn:kv_h2d"), torch.cuda.stream(workspace.h2d_stream):
            if workspace.kv_has_pending_compute[buffer_index]:
                workspace.h2d_stream.wait_event(workspace.kv_free[buffer_index])
            destination_k.copy_(
                self.k_cpu[start:stop],
                non_blocking=self.k_cpu.is_pinned(),
            )
            destination_v.copy_(
                self.v_cpu[start:stop],
                non_blocking=self.v_cpu.is_pinned(),
            )
            workspace.kv_ready[buffer_index].record(workspace.h2d_stream)
        compute_stream.wait_event(workspace.kv_ready[buffer_index])
        stats.h2d_bytes += (
            2 * tokens * self.k_cpu.shape[1] * self.k_cpu.shape[2] * self.k_cpu.element_size()
        )

    def release_q(self, compute_stream: torch.cuda.Stream) -> None:
        self.workspace.q_free.record(compute_stream)
        self.workspace.q_has_pending_compute = True

    def release_kv(self, buffer_index: int, compute_stream: torch.cuda.Stream) -> None:
        self.workspace.kv_free[buffer_index].record(compute_stream)
        self.workspace.kv_has_pending_compute[buffer_index] = True


class RecomputedQKVTileSource:
    def __init__(
        self,
        hidden_cpu: torch.Tensor,
        workspace: RecomputeWorkspace,
        *,
        project_q: QTileProjector,
        project_kv: KVTileProjector,
        stats: RecomputedAttentionStats,
        enable_nvtx: bool,
    ) -> None:
        self.hidden_cpu = hidden_cpu
        self.workspace = workspace
        self.project_q = project_q
        self.project_kv = project_kv
        self.stats = stats
        self.enable_nvtx = enable_nvtx

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.enable_nvtx else nullcontext()

    def _stage_hidden(
        self,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        workspace = self.workspace
        tokens = stop - start
        with self._range("seqattn:recompute_hidden_h2d"), torch.cuda.stream(workspace.h2d_stream):
            if workspace.hidden_has_pending_compute:
                workspace.h2d_stream.wait_event(workspace.hidden_free)
            workspace.hidden[:tokens].copy_(
                self.hidden_cpu[start:stop],
                non_blocking=self.hidden_cpu.is_pinned(),
            )
            workspace.hidden_ready.record(workspace.h2d_stream)
        compute_stream.wait_event(workspace.hidden_ready)
        self.stats.hidden_h2d_bytes += (
            tokens * self.hidden_cpu.shape[1] * self.hidden_cpu.element_size()
        )
        return workspace.hidden[:tokens]

    def _release_hidden(self, compute_stream: torch.cuda.Stream) -> None:
        self.workspace.hidden_free.record(compute_stream)
        self.workspace.hidden_has_pending_compute = True

    def load_q(
        self,
        destination: torch.Tensor,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None:
        del stats
        hidden = self._stage_hidden(start, stop, compute_stream)
        with self._range("seqattn:recompute_q_projection"):
            self.project_q(hidden, destination, start, stop)
        self._release_hidden(compute_stream)
        self.stats.q_projection_chunks += 1

    def load_kv(
        self,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        buffer_index: int,
        start: int,
        stop: int,
        compute_stream: torch.cuda.Stream,
        stats: StreamingAttentionStats,
    ) -> None:
        del buffer_index, stats
        hidden = self._stage_hidden(start, stop, compute_stream)
        with self._range("seqattn:recompute_kv_projection"):
            self.project_kv(hidden, destination_k, destination_v, start, stop)
        self._release_hidden(compute_stream)
        self.stats.kv_projection_chunks += 1

    def release_q(self, compute_stream: torch.cuda.Stream) -> None:
        del compute_stream

    def release_kv(self, buffer_index: int, compute_stream: torch.cuda.Stream) -> None:
        del buffer_index, compute_stream


__all__ = ["HostQKVTileSource", "QKVTileSource", "RecomputedQKVTileSource"]
