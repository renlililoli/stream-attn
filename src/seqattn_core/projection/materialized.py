from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext

import torch

from ..config import ProjectionPipelineConfig
from ..plan import AttentionPlan
from ..stats import ProjectedAttentionStats, ProjectedCrossAttentionStats
from .contracts import KVProjector, QKVProjector, QProjector
from .memory import MaterializedQKVArena, ProjectionWorkspace
from .validation import (
    validate_projected_kv,
    validate_projected_q,
    validate_projected_qkv,
)

ProjectedTile = tuple[torch.Tensor, ...]
MaterializedStats = ProjectedAttentionStats | ProjectedCrossAttentionStats


class MaterializedProjectionProducer:
    """Shared asynchronous hidden-to-host-QKV producer for self and cross attention."""

    def __init__(
        self,
        plan: AttentionPlan,
        config: ProjectionPipelineConfig,
        arena: MaterializedQKVArena,
    ) -> None:
        self.plan = plan
        self.config = config
        self.arena = arena
        self.self_workspace: ProjectionWorkspace | None = None
        self.query_workspace: ProjectionWorkspace | None = None
        self.context_workspace: ProjectionWorkspace | None = None

    def _range(self, name: str):
        return torch.cuda.nvtx.range(name) if self.config.enable_nvtx else nullcontext()

    def _workspace_for(
        self,
        workspace: ProjectionWorkspace | None,
        hidden_features: int,
        source_name: str,
    ) -> ProjectionWorkspace:
        if workspace is None:
            workspace = ProjectionWorkspace(
                hidden_features=hidden_features,
                dtype=self.plan.dtype,
                device=self.plan.device,
                config=self.config,
            )
        elif workspace.hidden_features != hidden_features:
            raise ValueError(f"{source_name} hidden feature size changed for a persistent runner")
        return workspace

    def _run_tiles(
        self,
        hidden_host: torch.Tensor,
        *,
        workspace: ProjectionWorkspace,
        h2d_range: str,
        projection_range: str,
        d2h_range: str,
        project: Callable[[torch.Tensor, int, int], ProjectedTile],
        validate: Callable[[ProjectedTile, int], None],
        copy_to_host: Callable[[int, int, ProjectedTile], None],
        record_chunk: Callable[[int], None],
        stats: MaterializedStats,
        ranges: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        tokens, hidden_features = hidden_host.shape
        chunk = self.config.projection_tile_tokens
        if ranges is None:
            ranges = ((start, min(start + chunk, tokens)) for start in range(0, tokens, chunk))
        started = time.perf_counter()
        expected_start = 0
        try:
            for chunk_index, (start, stop) in enumerate(ranges):
                if start != expected_start or not start < stop <= tokens:
                    raise ValueError("projection ranges must cover tokens contiguously in order")
                tile_tokens = stop - start
                if tile_tokens > chunk:
                    raise ValueError("a projection range exceeds projection_tile_tokens")
                expected_start = stop
                slot = chunk_index % len(workspace.hidden)
                if workspace.busy[slot]:
                    workspace.copy_done[slot].synchronize()
                    workspace.release_slot(slot)

                with self._range(h2d_range), torch.cuda.stream(workspace.h2d_stream):
                    workspace.hidden[slot][:tile_tokens].copy_(
                        hidden_host[start:stop], non_blocking=hidden_host.is_pinned()
                    )
                    workspace.input_ready[slot].record(workspace.h2d_stream)

                with torch.cuda.stream(workspace.compute_stream):
                    workspace.compute_stream.wait_event(workspace.input_ready[slot])
                    with self._range(projection_range):
                        projected = project(workspace.hidden[slot][:tile_tokens], start, stop)
                        validate(projected, tile_tokens)
                    workspace.projected_ready[slot].record(workspace.compute_stream)

                with self._range(d2h_range), torch.cuda.stream(workspace.d2h_stream):
                    workspace.d2h_stream.wait_event(workspace.projected_ready[slot])
                    copy_to_host(start, stop, projected)
                    workspace.copy_done[slot].record(workspace.d2h_stream)

                workspace.keepalive[slot] = projected
                workspace.busy[slot] = True
                record_chunk(tile_tokens)
                stats.projection_hidden_h2d_bytes += (
                    tile_tokens * hidden_features * hidden_host.element_size()
                )
                stats.projection_qkv_d2h_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in projected
                )
            if expected_start != tokens:
                raise ValueError("projection ranges must cover every input token")
            workspace.d2h_stream.synchronize()
        except Exception:
            workspace.recover()
            raise
        else:
            workspace.reset_slots()
        stats.projection_seconds += time.perf_counter() - started

    def project_qkv(
        self,
        hidden_host: torch.Tensor,
        project_qkv: QKVProjector,
        stats: ProjectedAttentionStats,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.self_workspace = self._workspace_for(
            self.self_workspace,
            hidden_host.shape[1],
            "self",
        )

        def project(tile: torch.Tensor, start: int, stop: int) -> ProjectedTile:
            q, k, v = project_qkv(tile, start, stop)
            return q, k, v

        def validate(projected: ProjectedTile, tokens: int) -> None:
            q, k, v = projected
            validate_projected_qkv(q, k, v, tokens=tokens, plan=self.plan)

        def copy_to_host(start: int, stop: int, projected: ProjectedTile) -> None:
            q, k, v = projected
            self.arena.q[start:stop].copy_(q, non_blocking=self.arena.q.is_pinned())
            self.arena.k[start:stop].copy_(k, non_blocking=self.arena.k.is_pinned())
            self.arena.v[start:stop].copy_(v, non_blocking=self.arena.v.is_pinned())

        def record_chunk(tokens: int) -> None:
            stats.projection_chunks += 1
            stats.projection_tokens += tokens

        self._run_tiles(
            hidden_host,
            workspace=self.self_workspace,
            h2d_range="seqattn:projection_hidden_h2d",
            projection_range="seqattn:qkv_projection",
            d2h_range="seqattn:projection_qkv_d2h",
            project=project,
            validate=validate,
            copy_to_host=copy_to_host,
            record_chunk=record_chunk,
            stats=stats,
        )
        return self.arena.views(hidden_host.shape[0], hidden_host.shape[0])

    def project_qkv_encoded(
        self,
        hidden_host: torch.Tensor,
        project_qkv: QKVProjector,
        *,
        ranges: Iterable[tuple[int, int]],
        encode: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, int, int],
            ProjectedTile,
        ],
        copy_to_host: Callable[[int, int, ProjectedTile], None],
        stats: ProjectedAttentionStats,
    ) -> None:
        """Project Q/K/V and send an encoded payload to caller-owned host storage."""

        self.self_workspace = self._workspace_for(
            self.self_workspace,
            hidden_host.shape[1],
            "self",
        )

        def project(tile: torch.Tensor, start: int, stop: int) -> ProjectedTile:
            q, k, v = project_qkv(tile, start, stop)
            validate_projected_qkv(q, k, v, tokens=stop - start, plan=self.plan)
            return encode(q, k, v, start, stop)

        def validate(projected: ProjectedTile, tokens: int) -> None:
            del tokens
            if not projected or any(not isinstance(tensor, torch.Tensor) for tensor in projected):
                raise TypeError("encoded Q/K/V payload must contain tensors")

        def record_chunk(tokens: int) -> None:
            stats.projection_chunks += 1
            stats.projection_tokens += tokens

        self._run_tiles(
            hidden_host,
            workspace=self.self_workspace,
            h2d_range="seqattn:projection_hidden_h2d",
            projection_range="seqattn:qkv_projection_encode",
            d2h_range="seqattn:projection_encoded_qkv_d2h",
            project=project,
            validate=validate,
            copy_to_host=copy_to_host,
            record_chunk=record_chunk,
            stats=stats,
            ranges=ranges,
        )

    def project_q(
        self,
        hidden_host: torch.Tensor,
        project_q: QProjector,
        stats: ProjectedCrossAttentionStats,
    ) -> torch.Tensor:
        self.query_workspace = self._workspace_for(
            self.query_workspace,
            hidden_host.shape[1],
            "query",
        )

        def project(tile: torch.Tensor, start: int, stop: int) -> ProjectedTile:
            return (project_q(tile, start, stop),)

        def validate(projected: ProjectedTile, tokens: int) -> None:
            (q,) = projected
            validate_projected_q(q, tokens=tokens, plan=self.plan)

        def copy_to_host(start: int, stop: int, projected: ProjectedTile) -> None:
            (q,) = projected
            self.arena.q[start:stop].copy_(q, non_blocking=self.arena.q.is_pinned())

        def record_chunk(tokens: int) -> None:
            stats.q_projection_chunks += 1
            stats.q_projection_tokens += tokens

        self._run_tiles(
            hidden_host,
            workspace=self.query_workspace,
            h2d_range="seqattn:cross_query_hidden_h2d",
            projection_range="seqattn:cross_q_projection",
            d2h_range="seqattn:cross_q_d2h",
            project=project,
            validate=validate,
            copy_to_host=copy_to_host,
            record_chunk=record_chunk,
            stats=stats,
        )
        return self.arena.q[: hidden_host.shape[0]]

    def project_kv(
        self,
        hidden_host: torch.Tensor,
        project_kv: KVProjector,
        stats: ProjectedCrossAttentionStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.context_workspace = self._workspace_for(
            self.context_workspace,
            hidden_host.shape[1],
            "context",
        )

        def project(tile: torch.Tensor, start: int, stop: int) -> ProjectedTile:
            k, v = project_kv(tile, start, stop)
            return k, v

        def validate(projected: ProjectedTile, tokens: int) -> None:
            k, v = projected
            validate_projected_kv(k, v, tokens=tokens, plan=self.plan)

        def copy_to_host(start: int, stop: int, projected: ProjectedTile) -> None:
            k, v = projected
            self.arena.k[start:stop].copy_(k, non_blocking=self.arena.k.is_pinned())
            self.arena.v[start:stop].copy_(v, non_blocking=self.arena.v.is_pinned())

        def record_chunk(tokens: int) -> None:
            stats.kv_projection_chunks += 1
            stats.kv_projection_tokens += tokens

        self._run_tiles(
            hidden_host,
            workspace=self.context_workspace,
            h2d_range="seqattn:cross_context_hidden_h2d",
            projection_range="seqattn:cross_kv_projection",
            d2h_range="seqattn:cross_kv_d2h",
            project=project,
            validate=validate,
            copy_to_host=copy_to_host,
            record_chunk=record_chunk,
            stats=stats,
        )
        tokens = hidden_host.shape[0]
        return self.arena.k[:tokens], self.arena.v[:tokens]


__all__ = ["MaterializedProjectionProducer"]
