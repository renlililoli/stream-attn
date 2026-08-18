from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Callable

import torch

from .config import ProjectionPipelineConfig, StreamingAttentionConfig
from .planner import AttentionPlan, build_plan
from .runtime import StreamingAttentionRunner
from .stats import ProjectedAttentionStats


QKVProjector = Callable[
    [torch.Tensor, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
OutputProjector = Callable[[torch.Tensor, int, int], torch.Tensor]


class _ProjectionWorkspace:
    def __init__(
        self,
        *,
        hidden_features: int,
        dtype: torch.dtype,
        device: torch.device,
        config: ProjectionPipelineConfig,
    ):
        self.hidden_features = hidden_features
        self.hidden = [
            torch.empty(
                (config.projection_chunk_tokens, hidden_features),
                dtype=dtype,
                device=device,
            )
            for _ in range(config.num_projection_buffers)
        ]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.input_ready = [torch.cuda.Event() for _ in self.hidden]
        self.projected_ready = [torch.cuda.Event() for _ in self.hidden]
        self.copy_done = [torch.cuda.Event() for _ in self.hidden]
        self.busy = [False for _ in self.hidden]
        self.keepalive: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None for _ in self.hidden
        ]


class ProjectedAttentionRunner:
    """Reusable CPU-hidden -> QKV -> attention -> output-projection pipeline.

    The QKV projector callback runs on GPU projection chunks and returns final
    Q/K/V tiles, including any model-specific Q/K normalization or positional
    transform.  Complete Q/K/V are stored in pinned CPU backing buffers because
    exact self-attention requires all K/V before a query result is final.

    During the second phase, each finalized attention tile remains on GPU and is
    passed directly to ``output_projector``.  Only the projected result is copied
    to CPU, eliminating the raw-attention D2H/H2D round trip.
    """

    def __init__(
        self,
        plan: AttentionPlan,
        attention_config: StreamingAttentionConfig | None = None,
        pipeline_config: ProjectionPipelineConfig | None = None,
    ):
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
        plan = self.plan
        if plan.device.type != "cuda":
            raise ValueError("the projected pipeline requires a CUDA device")
        if self.attention_config.backend not in {"auto", "triton"}:
            raise ValueError("the projected pipeline requires the Triton attention backend")
        if self.attention_config.require_pinned and not self.pipeline_config.pin_qkv:
            raise ValueError("Triton attention requires pinned Q/K/V backing buffers")

        self.attention = StreamingAttentionRunner(plan, self.attention_config)
        if self.attention.backend != "triton":
            raise RuntimeError("Triton is not available for the projected pipeline")
        pin_qkv = self.pipeline_config.pin_qkv
        self.q_cpu = torch.empty(
            (plan.max_q_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self.k_cpu = torch.empty(
            (plan.max_kv_tokens, plan.kv_heads, plan.head_dim),
            dtype=plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self.v_cpu = torch.empty(
            (plan.max_kv_tokens, plan.kv_heads, plan.head_dim),
            dtype=plan.dtype,
            device="cpu",
            pin_memory=pin_qkv,
        )
        self._projection_workspace: _ProjectionWorkspace | None = None

    def _range(self, name: str):
        if self.pipeline_config.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _workspace_for(self, hidden_features: int) -> _ProjectionWorkspace:
        if self._projection_workspace is None:
            self._projection_workspace = _ProjectionWorkspace(
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

    def _validate_hidden(self, hidden_cpu: torch.Tensor) -> None:
        if hidden_cpu.device.type != "cpu" or hidden_cpu.ndim != 2:
            raise ValueError("hidden_cpu must use CPU [tokens, hidden_features] layout")
        if not hidden_cpu.is_contiguous():
            raise ValueError("hidden_cpu must be contiguous")
        if hidden_cpu.dtype != self.plan.dtype:
            raise ValueError("hidden_cpu dtype must match the attention plan")
        if self.pipeline_config.require_pinned_hidden and not hidden_cpu.is_pinned():
            raise ValueError("asynchronous projection requires pinned hidden_cpu")
        tokens = hidden_cpu.shape[0]
        if tokens > self.plan.max_q_tokens or tokens > self.plan.max_kv_tokens:
            raise ValueError("hidden token count exceeds the runner plan")

    def _validate_projected_tile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tokens: int,
    ) -> None:
        expected_q = (tokens, self.plan.q_heads, self.plan.head_dim)
        expected_kv = (tokens, self.plan.kv_heads, self.plan.head_dim)
        if q.shape != expected_q:
            raise ValueError(
                f"project_qkv returned q shape {tuple(q.shape)}, expected {expected_q}"
            )
        if k.shape != expected_kv or v.shape != expected_kv:
            raise ValueError(
                "project_qkv returned invalid k/v shapes: "
                f"{tuple(k.shape)}, {tuple(v.shape)}, expected {expected_kv}"
            )
        if any(t.device != self.plan.device for t in (q, k, v)):
            raise ValueError("project_qkv must return tensors on the planned CUDA device")
        if any(t.dtype != self.plan.dtype for t in (q, k, v)):
            raise ValueError("project_qkv output dtype must match the attention plan")

    def project_qkv_to_host(
        self,
        hidden_cpu: torch.Tensor,
        project_qkv: QKVProjector,
        stats: ProjectedAttentionStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the chunked projection producer and return CPU-backed Q/K/V views."""

        stats = ProjectedAttentionStats() if stats is None else stats
        self._validate_hidden(hidden_cpu)
        stats.backend = self.attention.backend
        tokens, hidden_features = hidden_cpu.shape
        workspace = self._workspace_for(hidden_features)
        chunk = self.pipeline_config.projection_chunk_tokens
        torch.cuda.synchronize(self.plan.device)
        started = time.perf_counter()

        for chunk_index, start in enumerate(range(0, tokens, chunk)):
            stop = min(start + chunk, tokens)
            tile_tokens = stop - start
            slot = chunk_index % len(workspace.hidden)
            if workspace.busy[slot]:
                workspace.copy_done[slot].synchronize()
                workspace.keepalive[slot] = None

            with self._range("seqattn:projection_hidden_h2d"):
                with torch.cuda.stream(workspace.h2d_stream):
                    workspace.hidden[slot][:tile_tokens].copy_(
                        hidden_cpu[start:stop], non_blocking=hidden_cpu.is_pinned()
                    )
                    workspace.input_ready[slot].record(workspace.h2d_stream)

            with torch.cuda.stream(workspace.compute_stream):
                workspace.compute_stream.wait_event(workspace.input_ready[slot])
                with self._range("seqattn:qkv_projection"):
                    q, k, v = project_qkv(
                        workspace.hidden[slot][:tile_tokens], start, stop
                    )
                    self._validate_projected_tile(q, k, v, tile_tokens)
                workspace.projected_ready[slot].record(workspace.compute_stream)

            with self._range("seqattn:projection_qkv_d2h"):
                with torch.cuda.stream(workspace.d2h_stream):
                    workspace.d2h_stream.wait_event(workspace.projected_ready[slot])
                    self.q_cpu[start:stop].copy_(q, non_blocking=self.q_cpu.is_pinned())
                    self.k_cpu[start:stop].copy_(k, non_blocking=self.k_cpu.is_pinned())
                    self.v_cpu[start:stop].copy_(v, non_blocking=self.v_cpu.is_pinned())
                    workspace.copy_done[slot].record(workspace.d2h_stream)

            workspace.keepalive[slot] = (q, k, v)
            workspace.busy[slot] = True
            stats.projection_chunks += 1
            stats.projection_hidden_h2d_bytes += (
                tile_tokens * hidden_features * hidden_cpu.element_size()
            )
            stats.projection_qkv_d2h_bytes += sum(
                tensor.numel() * tensor.element_size() for tensor in (q, k, v)
            )

        workspace.d2h_stream.synchronize()
        for slot in range(len(workspace.keepalive)):
            workspace.keepalive[slot] = None
        stats.projection_seconds += time.perf_counter() - started
        q_cpu = self.q_cpu[:tokens]
        k_cpu = self.k_cpu[:tokens]
        v_cpu = self.v_cpu[:tokens]
        stats.qkv_host_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
        )
        return q_cpu, k_cpu, v_cpu

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
        self._validate_hidden(hidden_cpu)
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


def streaming_projected_self_attention(
    hidden_cpu: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    project_qkv: QKVProjector,
    output_projector: OutputProjector,
    output_features: int,
    attention_config: StreamingAttentionConfig | None = None,
    pipeline_config: ProjectionPipelineConfig | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    stats: ProjectedAttentionStats | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Functional convenience wrapper for one projected self-attention call."""

    attention_config = (
        StreamingAttentionConfig() if attention_config is None else attention_config
    )
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=hidden_cpu.dtype,
        device=device,
        max_q_tokens=hidden_cpu.shape[0],
        max_kv_tokens=hidden_cpu.shape[0],
        config=attention_config,
    )
    runner = ProjectedAttentionRunner(plan, attention_config, pipeline_config)
    return runner(
        hidden_cpu,
        cu_seqlens,
        project_qkv=project_qkv,
        output_projector=output_projector,
        output_features=output_features,
        softmax_scale=softmax_scale,
        causal=causal,
        stats=stats,
    )


__all__ = [
    "OutputProjector",
    "ProjectedAttentionRunner",
    "QKVProjector",
    "streaming_projected_self_attention",
]
