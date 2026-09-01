from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from .._single_flight import init_single_flight, single_flight
from ..config import ProjectionPipelineConfig, StreamingAttentionConfig
from ..planner import AttentionPlan
from ..stats import ProjectedAttentionStats
from ..streaming import StreamingAttentionRunner
from .arena import MaterializedQKVArena
from .types import OutputProjector, QKVProjector
from .validation import validate_projected_qkv, validate_projection_hidden
from .workspace import ProjectionWorkspace


class ProjectedAttentionRunner:
    """Reusable CPU-hidden -> QKV -> attention -> output-projection pipeline."""

    def __init__(
        self,
        plan: AttentionPlan,
        attention_config: StreamingAttentionConfig | None = None,
        pipeline_config: ProjectionPipelineConfig | None = None,
        *,
        arena: MaterializedQKVArena | None = None,
    ) -> None:
        init_single_flight(self)
        self.pipeline_config = (
            ProjectionPipelineConfig() if pipeline_config is None else pipeline_config
        )
        self.pipeline_config.validate()
        self.plan = plan
        if plan.device.type != "cuda":
            raise ValueError("the projected pipeline requires a CUDA device")
        self.attention = StreamingAttentionRunner(plan, attention_config)
        self.attention_config = self.attention.config
        if self.attention_config.backend not in {None, "auto", "builtin", "triton"}:
            raise ValueError("the projected pipeline requires the Triton attention backend")
        if self.attention_config.require_pinned and not self.pipeline_config.pin_qkv:
            raise ValueError("Triton attention requires pinned Q/K/V backing buffers")

        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for the projected pipeline")
        pin_qkv = self.pipeline_config.pin_qkv
        self.arena = (
            MaterializedQKVArena.for_plans((plan,), pin_memory=pin_qkv) if arena is None else arena
        )
        self.arena.validate_plan(plan)
        if self.arena.pin_memory != pin_qkv:
            raise ValueError("QKV arena pinning must match the projection pipeline")
        self._projection_workspace: ProjectionWorkspace | None = None

    def _range(self, name: str):
        if self.pipeline_config.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _workspace_for(self, hidden_features: int) -> ProjectionWorkspace:
        if self._projection_workspace is None:
            self._projection_workspace = ProjectionWorkspace(
                hidden_features=hidden_features,
                dtype=self.plan.dtype,
                device=self.plan.device,
                config=self.pipeline_config,
            )
        elif self._projection_workspace.hidden_features != hidden_features:
            raise ValueError(
                "hidden feature size changed for a persistent ProjectedAttentionRunner"
            )
        return self._projection_workspace

    @single_flight
    def project_qkv_to_host(
        self,
        hidden_cpu: torch.Tensor,
        project_qkv: QKVProjector,
        stats: ProjectedAttentionStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the chunked projection producer and return CPU-backed Q/K/V views."""

        stats = ProjectedAttentionStats() if stats is None else stats
        validate_projection_hidden(
            hidden_cpu,
            plan=self.plan,
            require_pinned=self.pipeline_config.require_pinned_hidden,
        )
        stats.backend = self.attention.backend
        tokens, hidden_features = hidden_cpu.shape
        workspace = self._workspace_for(hidden_features)
        chunk = self.pipeline_config.projection_tile_tokens
        started = time.perf_counter()
        try:
            for chunk_index, start in enumerate(range(0, tokens, chunk)):
                stop = min(start + chunk, tokens)
                tile_tokens = stop - start
                slot = chunk_index % len(workspace.hidden)
                if workspace.busy[slot]:
                    workspace.copy_done[slot].synchronize()
                    workspace.release_slot(slot)

                with (
                    self._range("seqattn:projection_hidden_h2d"),
                    torch.cuda.stream(workspace.h2d_stream),
                ):
                    workspace.hidden[slot][:tile_tokens].copy_(
                        hidden_cpu[start:stop], non_blocking=hidden_cpu.is_pinned()
                    )
                    workspace.input_ready[slot].record(workspace.h2d_stream)

                with torch.cuda.stream(workspace.compute_stream):
                    workspace.compute_stream.wait_event(workspace.input_ready[slot])
                    with self._range("seqattn:qkv_projection"):
                        q, k, v = project_qkv(workspace.hidden[slot][:tile_tokens], start, stop)
                        validate_projected_qkv(q, k, v, tokens=tile_tokens, plan=self.plan)
                    workspace.projected_ready[slot].record(workspace.compute_stream)

                with (
                    self._range("seqattn:projection_qkv_d2h"),
                    torch.cuda.stream(workspace.d2h_stream),
                ):
                    workspace.d2h_stream.wait_event(workspace.projected_ready[slot])
                    self.arena.q[start:stop].copy_(q, non_blocking=self.arena.q.is_pinned())
                    self.arena.k[start:stop].copy_(k, non_blocking=self.arena.k.is_pinned())
                    self.arena.v[start:stop].copy_(v, non_blocking=self.arena.v.is_pinned())
                    workspace.copy_done[slot].record(workspace.d2h_stream)

                workspace.keepalive[slot] = (q, k, v)
                workspace.busy[slot] = True
                stats.projection_chunks += 1
                stats.projection_tokens += tile_tokens
                stats.projection_hidden_h2d_bytes += (
                    tile_tokens * hidden_features * hidden_cpu.element_size()
                )
                stats.projection_qkv_d2h_bytes += sum(
                    tensor.numel() * tensor.element_size() for tensor in (q, k, v)
                )

            workspace.d2h_stream.synchronize()
        except Exception:
            workspace.recover()
            raise
        else:
            workspace.reset_slots()
        stats.projection_seconds += time.perf_counter() - started
        q_cpu, k_cpu, v_cpu = self.arena.views(tokens, tokens)
        stats.qkv_host_bytes = self.arena.allocated_bytes
        return q_cpu, k_cpu, v_cpu

    @single_flight
    @torch.inference_mode()
    def __call__(
        self,
        hidden_cpu: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        project_qkv: QKVProjector,
        output_projector: OutputProjector,
        output_features: int | None = None,
        out: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: ProjectedAttentionStats | None = None,
    ) -> torch.Tensor:
        validate_projection_hidden(
            hidden_cpu,
            plan=self.plan,
            require_pinned=self.pipeline_config.require_pinned_hidden,
        )
        tokens = hidden_cpu.shape[0]
        if out is None:
            if output_features is None or output_features <= 0:
                raise ValueError("output_features must be positive when out is omitted")
            out = torch.empty(
                (tokens, output_features),
                dtype=self.plan.dtype,
                device="cpu",
                pin_memory=self.pipeline_config.pin_output,
            )
        if out.device.type != "cpu" or out.ndim != 2 or out.shape[0] != tokens:
            raise ValueError("out must use CPU [tokens, output_features] layout")
        if out.dtype != self.plan.dtype:
            raise ValueError("out dtype must match the attention plan")
        if self.pipeline_config.pin_output and not out.is_pinned():
            raise ValueError("asynchronous output D2H requires pinned out")

        stats = ProjectedAttentionStats() if stats is None else stats
        stats.backend = self.attention.backend
        started = time.perf_counter()
        q_cpu, k_cpu, v_cpu = self.project_qkv_to_host(hidden_cpu, project_qkv, stats)
        raw_output_bytes = (
            tokens
            * self.plan.q_heads
            * self.plan.head_dim
            * torch.empty((), dtype=self.plan.dtype).element_size()
        )
        stats.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes

        attention_started = time.perf_counter()
        self.attention.run_with_device_output(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens,
            cu_seqlens,
            output_transform=output_projector,
            out=out,
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats.attention,
        )
        stats.attention_output_seconds += time.perf_counter() - attention_started
        stats.wall_seconds += time.perf_counter() - started
        return out


__all__ = ["ProjectedAttentionRunner"]
