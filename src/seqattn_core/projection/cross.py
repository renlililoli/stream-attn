from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from ..config import ProjectionPipelineConfig, StreamingAttentionConfig
from ..planner import AttentionPlan, build_plan
from ..stats import ProjectedCrossAttentionStats
from ..streaming import StreamingAttentionRunner
from .types import KVProjector, OutputProjector, QProjector
from .workspace import ProjectionWorkspace


class ProjectedCrossAttentionRunner:
    """Project independent query/context hidden states into exact cross-attention."""

    def __init__(
        self,
        plan: AttentionPlan,
        attention_config: StreamingAttentionConfig | None = None,
        pipeline_config: ProjectionPipelineConfig | None = None,
    ) -> None:
        self.attention_config = (
            StreamingAttentionConfig() if attention_config is None else attention_config
        )
        self.pipeline_config = (
            ProjectionPipelineConfig() if pipeline_config is None else pipeline_config
        )
        self.attention_config.validate()
        self.pipeline_config.validate()
        self.plan = build_plan(
            q_heads=plan.q_heads,
            kv_heads=plan.kv_heads,
            head_dim=plan.head_dim,
            dtype=plan.dtype,
            device=plan.device,
            max_q_tokens=plan.max_q_tokens,
            max_kv_tokens=plan.max_kv_tokens,
            config=self.attention_config,
        )
        if self.plan.device.type != "cuda":
            raise ValueError("the projected cross-attention pipeline requires a CUDA device")
        if self.attention_config.backend not in {None, "auto", "builtin", "triton"}:
            raise ValueError("projected cross-attention requires the Triton attention backend")
        if self.attention_config.require_pinned and not self.pipeline_config.pin_qkv:
            raise ValueError("Triton attention requires pinned Q/K/V backing buffers")

        self.attention = StreamingAttentionRunner(self.plan, self.attention_config)
        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for projected cross-attention")
        pin_qkv = self.pipeline_config.pin_qkv
        self.q_cpu = torch.empty(
            (self.plan.max_q_tokens, self.plan.q_heads, self.plan.head_dim),
            dtype=self.plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self.k_cpu = torch.empty(
            (self.plan.max_kv_tokens, self.plan.kv_heads, self.plan.head_dim),
            dtype=self.plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self.v_cpu = torch.empty(
            self.k_cpu.shape,
            dtype=self.plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self._query_workspace: ProjectionWorkspace | None = None
        self._context_workspace: ProjectionWorkspace | None = None

    def _range(self, name: str):
        if self.pipeline_config.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _workspace_for(self, *, query: bool, hidden_features: int) -> ProjectionWorkspace:
        attribute = "_query_workspace" if query else "_context_workspace"
        workspace = getattr(self, attribute)
        if workspace is None:
            workspace = ProjectionWorkspace(
                hidden_features=hidden_features,
                dtype=self.plan.dtype,
                device=self.plan.device,
                config=self.pipeline_config,
            )
            setattr(self, attribute, workspace)
        elif workspace.hidden_features != hidden_features:
            source = "query" if query else "context"
            raise ValueError(f"{source} hidden feature size changed for a persistent runner")
        return workspace

    def _validate_hidden(
        self,
        hidden_host: torch.Tensor,
        *,
        max_tokens: int,
        name: str,
    ) -> None:
        if hidden_host.device.type != "cpu" or hidden_host.ndim != 2:
            raise ValueError(f"{name} must use CPU [tokens, hidden_features] layout")
        if not hidden_host.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if hidden_host.dtype != self.plan.dtype:
            raise ValueError(f"{name} dtype must match the attention plan")
        if self.pipeline_config.require_pinned_hidden and not hidden_host.is_pinned():
            raise ValueError(f"asynchronous projection requires pinned {name}")
        if hidden_host.shape[0] > max_tokens:
            raise ValueError(f"{name} token count exceeds the runner plan")

    def validate_inputs(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
    ) -> None:
        self._validate_hidden(
            query_hidden_host,
            max_tokens=self.plan.max_q_tokens,
            name="query_hidden_host",
        )
        self._validate_hidden(
            context_hidden_host,
            max_tokens=self.plan.max_kv_tokens,
            name="context_hidden_host",
        )

    def _project_q_to_host(
        self,
        hidden_host: torch.Tensor,
        project_q: QProjector,
        stats: ProjectedCrossAttentionStats,
    ) -> torch.Tensor:
        tokens, hidden_features = hidden_host.shape
        workspace = self._workspace_for(query=True, hidden_features=hidden_features)
        chunk = self.pipeline_config.projection_chunk_tokens
        started = time.perf_counter()
        for chunk_index, start in enumerate(range(0, tokens, chunk)):
            stop = min(start + chunk, tokens)
            tile_tokens = stop - start
            slot = chunk_index % len(workspace.hidden)
            if workspace.busy[slot]:
                workspace.copy_done[slot].synchronize()
                workspace.keepalive[slot] = None
            with (
                self._range("seqattn:cross_query_hidden_h2d"),
                torch.cuda.stream(workspace.h2d_stream),
            ):
                workspace.hidden[slot][:tile_tokens].copy_(
                    hidden_host[start:stop],
                    non_blocking=hidden_host.is_pinned(),
                )
                workspace.input_ready[slot].record(workspace.h2d_stream)
            with torch.cuda.stream(workspace.compute_stream):
                workspace.compute_stream.wait_event(workspace.input_ready[slot])
                with self._range("seqattn:cross_q_projection"):
                    q = project_q(workspace.hidden[slot][:tile_tokens], start, stop)
                    expected = (tile_tokens, self.plan.q_heads, self.plan.head_dim)
                    if q.shape != expected:
                        raise ValueError(
                            f"project_q returned shape {tuple(q.shape)}, expected {expected}"
                        )
                    if q.device != self.plan.device or q.dtype != self.plan.dtype:
                        raise ValueError("project_q must return the planned CUDA dtype/device")
                workspace.projected_ready[slot].record(workspace.compute_stream)
            with self._range("seqattn:cross_q_d2h"), torch.cuda.stream(workspace.d2h_stream):
                workspace.d2h_stream.wait_event(workspace.projected_ready[slot])
                self.q_cpu[start:stop].copy_(q, non_blocking=self.q_cpu.is_pinned())
                workspace.copy_done[slot].record(workspace.d2h_stream)
            workspace.keepalive[slot] = q
            workspace.busy[slot] = True
            stats.q_projection_chunks += 1
            stats.q_projection_tokens += tile_tokens
            stats.projection_hidden_h2d_bytes += (
                hidden_host[start:stop].numel() * hidden_host.element_size()
            )
            stats.projection_qkv_d2h_bytes += q.numel() * q.element_size()
        workspace.d2h_stream.synchronize()
        workspace.keepalive[:] = [None] * len(workspace.keepalive)
        stats.projection_seconds += time.perf_counter() - started
        return self.q_cpu[:tokens]

    def _project_kv_to_host(
        self,
        hidden_host: torch.Tensor,
        project_kv: KVProjector,
        stats: ProjectedCrossAttentionStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, hidden_features = hidden_host.shape
        workspace = self._workspace_for(query=False, hidden_features=hidden_features)
        chunk = self.pipeline_config.projection_chunk_tokens
        started = time.perf_counter()
        for chunk_index, start in enumerate(range(0, tokens, chunk)):
            stop = min(start + chunk, tokens)
            tile_tokens = stop - start
            slot = chunk_index % len(workspace.hidden)
            if workspace.busy[slot]:
                workspace.copy_done[slot].synchronize()
                workspace.keepalive[slot] = None
            with (
                self._range("seqattn:cross_context_hidden_h2d"),
                torch.cuda.stream(workspace.h2d_stream),
            ):
                workspace.hidden[slot][:tile_tokens].copy_(
                    hidden_host[start:stop],
                    non_blocking=hidden_host.is_pinned(),
                )
                workspace.input_ready[slot].record(workspace.h2d_stream)
            with torch.cuda.stream(workspace.compute_stream):
                workspace.compute_stream.wait_event(workspace.input_ready[slot])
                with self._range("seqattn:cross_kv_projection"):
                    k, v = project_kv(workspace.hidden[slot][:tile_tokens], start, stop)
                    expected = (tile_tokens, self.plan.kv_heads, self.plan.head_dim)
                    if k.shape != expected or v.shape != expected:
                        raise ValueError(
                            "project_kv returned invalid shapes: "
                            f"{tuple(k.shape)}, {tuple(v.shape)}, expected {expected}"
                        )
                    if any(t.device != self.plan.device for t in (k, v)) or any(
                        t.dtype != self.plan.dtype for t in (k, v)
                    ):
                        raise ValueError("project_kv must return the planned CUDA dtype/device")
                workspace.projected_ready[slot].record(workspace.compute_stream)
            with self._range("seqattn:cross_kv_d2h"), torch.cuda.stream(workspace.d2h_stream):
                workspace.d2h_stream.wait_event(workspace.projected_ready[slot])
                self.k_cpu[start:stop].copy_(k, non_blocking=self.k_cpu.is_pinned())
                self.v_cpu[start:stop].copy_(v, non_blocking=self.v_cpu.is_pinned())
                workspace.copy_done[slot].record(workspace.d2h_stream)
            workspace.keepalive[slot] = (k, v)
            workspace.busy[slot] = True
            stats.kv_projection_chunks += 1
            stats.kv_projection_tokens += tile_tokens
            stats.projection_hidden_h2d_bytes += (
                hidden_host[start:stop].numel() * hidden_host.element_size()
            )
            stats.projection_qkv_d2h_bytes += (k.numel() + v.numel()) * k.element_size()
        workspace.d2h_stream.synchronize()
        workspace.keepalive[:] = [None] * len(workspace.keepalive)
        stats.projection_seconds += time.perf_counter() - started
        return self.k_cpu[:tokens], self.v_cpu[:tokens]

    def project_to_host(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        *,
        project_q: QProjector,
        project_kv: KVProjector,
        stats: ProjectedCrossAttentionStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_inputs(query_hidden_host, context_hidden_host)
        stats = ProjectedCrossAttentionStats() if stats is None else stats
        stats.backend = self.attention.backend
        q_cpu = self._project_q_to_host(query_hidden_host, project_q, stats)
        k_cpu, v_cpu = self._project_kv_to_host(context_hidden_host, project_kv, stats)
        stats.qkv_host_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
        )
        return q_cpu, k_cpu, v_cpu

    @torch.inference_mode()
    def run_with_device_consumer(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        *,
        project_q: QProjector,
        project_kv: KVProjector,
        output_consumer,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: ProjectedCrossAttentionStats | None = None,
    ) -> None:
        stats = ProjectedCrossAttentionStats() if stats is None else stats
        started = time.perf_counter()
        q_cpu, k_cpu, v_cpu = self.project_to_host(
            query_hidden_host,
            context_hidden_host,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
        )
        raw_output_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        attention_started = time.perf_counter()
        self.attention.run_with_device_consumer(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens_q,
            cu_seqlens_kv,
            output_consumer=output_consumer,
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats.attention,
        )
        stats.attention_output_seconds += time.perf_counter() - attention_started
        stats.wall_seconds += time.perf_counter() - started

    @torch.inference_mode()
    def __call__(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        *,
        project_q: QProjector,
        project_kv: KVProjector,
        output_projector: OutputProjector,
        output_features: int | None = None,
        out: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: ProjectedCrossAttentionStats | None = None,
    ) -> torch.Tensor:
        self.validate_inputs(query_hidden_host, context_hidden_host)
        query_tokens = query_hidden_host.shape[0]
        if out is None:
            if output_features is None or output_features <= 0:
                raise ValueError("output_features must be positive when out is omitted")
            out = torch.empty(
                (query_tokens, output_features),
                dtype=self.plan.dtype,
                device="cpu",
                pin_memory=self.pipeline_config.pin_output,
            )
        if out.device.type != "cpu" or out.ndim != 2 or out.shape[0] != query_tokens:
            raise ValueError("out must use CPU [query_tokens, output_features] layout")
        if out.dtype != self.plan.dtype:
            raise ValueError("out dtype must match the attention plan")
        if self.pipeline_config.pin_output and not out.is_pinned():
            raise ValueError("asynchronous output D2H requires pinned out")

        stats = ProjectedCrossAttentionStats() if stats is None else stats
        started = time.perf_counter()
        q_cpu, k_cpu, v_cpu = self.project_to_host(
            query_hidden_host,
            context_hidden_host,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
        )
        raw_output_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        attention_started = time.perf_counter()
        self.attention.run_with_device_output(
            q_cpu,
            k_cpu,
            v_cpu,
            cu_seqlens_q,
            cu_seqlens_kv,
            output_transform=output_projector,
            out=out,
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats.attention,
        )
        stats.attention_output_seconds += time.perf_counter() - attention_started
        stats.wall_seconds += time.perf_counter() - started
        return out


__all__ = ["ProjectedCrossAttentionRunner"]
