from __future__ import annotations

import time

import torch

from .._single_flight import init_single_flight, single_flight
from ..config import ProjectionPipelineConfig
from ..plan import AttentionPlan
from ..stats import ProjectedAttentionStats, ProjectedCrossAttentionStats
from ..streaming import StreamingAttentionRunner
from .contracts import KVProjector, OutputProjector, QKVProjector, QProjector
from .materialized import MaterializedProjectionProducer
from .memory import MaterializedQKVArena, ProjectionWorkspace
from .validation import validate_projection_hidden


class _MaterializedAttentionRunner:
    def __init__(
        self,
        plan: AttentionPlan,
        pipeline_config: ProjectionPipelineConfig | None,
        arena: MaterializedQKVArena | None,
    ) -> None:
        init_single_flight(self)
        self.pipeline_config = (
            ProjectionPipelineConfig() if pipeline_config is None else pipeline_config
        )
        self.pipeline_config.validate()
        self.plan = plan
        if plan.device.type != "cuda":
            raise ValueError("the projected pipeline requires a CUDA device")
        self.attention = StreamingAttentionRunner(plan)
        if plan.backend not in {"auto", "triton"}:
            raise ValueError("the projected pipeline requires the Triton attention backend")
        if plan.require_pinned and not self.pipeline_config.pin_qkv:
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
        self._producer = MaterializedProjectionProducer(plan, self.pipeline_config, self.arena)

    def _prepare_output(
        self,
        tokens: int,
        output_features: int | None,
        out: torch.Tensor | None,
        *,
        token_label: str,
    ) -> torch.Tensor:
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
            raise ValueError(f"out must use CPU [{token_label}, output_features] layout")
        if out.dtype != self.plan.dtype:
            raise ValueError("out dtype must match the attention plan")
        if self.pipeline_config.pin_output and not out.is_pinned():
            raise ValueError("asynchronous output D2H requires pinned out")
        return out


class ProjectedAttentionRunner(_MaterializedAttentionRunner):
    """Reusable CPU-hidden -> QKV -> attention -> output-projection pipeline."""

    def __init__(
        self,
        plan: AttentionPlan,
        pipeline_config: ProjectionPipelineConfig | None = None,
        *,
        arena: MaterializedQKVArena | None = None,
    ) -> None:
        super().__init__(plan, pipeline_config, arena)

    @property
    def _projection_workspace(self) -> ProjectionWorkspace | None:
        return self._producer.self_workspace

    @single_flight
    def project_qkv_to_host(
        self,
        hidden_cpu: torch.Tensor,
        project_qkv: QKVProjector,
        stats: ProjectedAttentionStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = ProjectedAttentionStats() if stats is None else stats
        validate_projection_hidden(
            hidden_cpu,
            plan=self.plan,
            require_pinned=self.pipeline_config.require_pinned_hidden,
        )
        stats.backend = self.attention.backend
        q, k, v = self._producer.project_qkv(hidden_cpu, project_qkv, stats)
        stats.qkv_host_bytes = self.arena.allocated_bytes
        return q, k, v

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
        out = self._prepare_output(tokens, output_features, out, token_label="tokens")
        stats = ProjectedAttentionStats() if stats is None else stats
        stats.backend = self.attention.backend
        started = time.perf_counter()
        q_cpu, k_cpu, v_cpu = self.project_qkv_to_host(hidden_cpu, project_qkv, stats)
        stats.raw_attention_roundtrip_bytes_avoided += 2 * q_cpu.numel() * q_cpu.element_size()
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


class ProjectedCrossAttentionRunner(_MaterializedAttentionRunner):
    """Project independent query/context hidden states into exact cross-attention."""

    def __init__(
        self,
        plan: AttentionPlan,
        pipeline_config: ProjectionPipelineConfig | None = None,
        *,
        query_hidden_features: int,
        context_hidden_features: int,
        arena: MaterializedQKVArena | None = None,
    ) -> None:
        if query_hidden_features <= 0 or context_hidden_features <= 0:
            raise ValueError("query and context hidden feature sizes must be positive")
        self.query_hidden_features = query_hidden_features
        self.context_hidden_features = context_hidden_features
        super().__init__(plan, pipeline_config, arena)

    def validate_hidden(
        self,
        hidden_host: torch.Tensor,
        *,
        hidden_features: int,
        max_tokens: int,
        name: str,
    ) -> None:
        validate_projection_hidden(
            hidden_host,
            plan=self.plan,
            require_pinned=self.pipeline_config.require_pinned_hidden,
            hidden_features=hidden_features,
            max_tokens=max_tokens,
            name=name,
        )

    def validate_inputs(
        self,
        query_hidden_host: torch.Tensor,
        context_hidden_host: torch.Tensor,
    ) -> None:
        self.validate_hidden(
            query_hidden_host,
            hidden_features=self.query_hidden_features,
            max_tokens=self.plan.max_q_tokens,
            name="query_hidden_host",
        )
        self.validate_hidden(
            context_hidden_host,
            hidden_features=self.context_hidden_features,
            max_tokens=self.plan.max_kv_tokens,
            name="context_hidden_host",
        )

    @single_flight
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
        q_cpu = self._producer.project_q(query_hidden_host, project_q, stats)
        k_cpu, v_cpu = self._producer.project_kv(context_hidden_host, project_kv, stats)
        stats.qkv_host_bytes = self.arena.allocated_bytes
        return q_cpu, k_cpu, v_cpu

    @single_flight
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
        stats.raw_attention_roundtrip_bytes_avoided += 2 * q_cpu.numel() * q_cpu.element_size()
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

    @single_flight
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
        out = self._prepare_output(query_tokens, output_features, out, token_label="query_tokens")
        stats = ProjectedCrossAttentionStats() if stats is None else stats
        started = time.perf_counter()
        q_cpu, k_cpu, v_cpu = self.project_to_host(
            query_hidden_host,
            context_hidden_host,
            project_q=project_q,
            project_kv=project_kv,
            stats=stats,
        )
        stats.raw_attention_roundtrip_bytes_avoided += 2 * q_cpu.numel() * q_cpu.element_size()
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


__all__ = ["ProjectedAttentionRunner", "ProjectedCrossAttentionRunner"]
