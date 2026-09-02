from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import ProjectedAttentionRunner
from ..common import validate_hidden_host
from .consumer import H3DeviceOutputConsumer
from .stats import H3DiTStats
from .types import (
    H3BlockOps,
    H3MaterializedPlan,
    H3MaterializedProjection,
    H3SequenceMeta,
    estimate_h3_materialized_aux_workspace_bytes,
)
from .workspace import H3BlockWorkspace


class H3MaterializedRunner:
    """MiniMax-H3 scheduler using sequence-sized host Q/K/V backing."""

    def __init__(
        self,
        projected_attention: ProjectedAttentionRunner,
        *,
        hidden_features: int,
        ffn_tile_tokens: int,
        num_final_output_buffers: int = 2,
    ) -> None:
        init_single_flight(self)
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if ffn_tile_tokens <= 0:
            raise ValueError("ffn_tile_tokens must be positive")
        if num_final_output_buffers not in {1, 2}:
            raise ValueError("num_final_output_buffers must be 1 or 2")
        if projected_attention.attention.backend != "triton":
            raise ValueError("the H3 fused block runner requires the Triton backend")
        if projected_attention.plan.output_mode != "device_consumer":
            raise ValueError("the H3 fused block runner requires device_consumer output mode")

        self.projected_attention = projected_attention
        self.hidden_features = hidden_features
        self.ffn_tile_tokens = ffn_tile_tokens
        attention_plan = projected_attention.plan
        pipeline_config = projected_attention.pipeline_config
        aux_workspace = estimate_h3_materialized_aux_workspace_bytes(
            hidden_features=hidden_features,
            dtype=attention_plan.dtype,
            projection_tile_tokens=pipeline_config.projection_tile_tokens,
            num_projection_buffers=pipeline_config.num_projection_buffers,
            ffn_tile_tokens=ffn_tile_tokens,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.plan = H3MaterializedPlan(
            projection_tile_tokens=pipeline_config.projection_tile_tokens,
            q_chunk_tokens=attention_plan.q_chunk_tokens,
            kv_chunk_tokens=attention_plan.kv_chunk_tokens,
            ffn_tile_tokens=ffn_tile_tokens,
            estimated_workspace_bytes=attention_plan.estimated_workspace_bytes + aux_workspace,
        )
        self.plan.validate()
        self.workspace = H3BlockWorkspace(
            hidden_features=hidden_features,
            ffn_tile_tokens=ffn_tile_tokens,
            dtype=attention_plan.dtype,
            device=attention_plan.device,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.consumer = H3DeviceOutputConsumer(self.workspace)

    def _validate_hidden(self, hidden_host: torch.Tensor) -> None:
        validate_hidden_host(
            hidden_host,
            plan=self.projected_attention.plan,
            hidden_features=self.hidden_features,
            require_pinned=self.projected_attention.pipeline_config.require_pinned_hidden,
        )

    @single_flight
    @torch.inference_mode()
    def run_block_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        projection: H3MaterializedProjection,
        ops: H3BlockOps,
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: H3DiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_hidden(hidden_host)
        sequence_meta.validate(hidden_host.shape[0])
        stats = H3DiTStats() if stats is None else stats
        stats.backend = self.projected_attention.attention.backend
        stats.qkv_storage_policy = "materialized"
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        started = time.perf_counter()

        with projection.context():
            q_cpu, k_cpu, v_cpu = self.projected_attention.project_qkv_to_host(
                hidden_host,
                projection.project_qkv,
                stats=stats.projection,
            )
        qkv_host_bytes = self.projected_attention.arena.allocated_bytes
        stats.qkv_host_bytes_peak = max(stats.qkv_host_bytes_peak, qkv_host_bytes)
        hidden_bytes = hidden_host.numel() * hidden_host.element_size()
        stats.hidden_host_bytes_peak = max(stats.hidden_host_bytes_peak, hidden_bytes)
        raw_attention_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.projection.raw_attention_roundtrip_bytes_avoided += 2 * raw_attention_bytes

        self.consumer.reset(
            destination_hidden_host=hidden_host,
            residual_hidden_host=hidden_host,
            ops=ops,
            stats=stats,
        )
        attention_started = time.perf_counter()
        with ops.consumer_context():
            self.projected_attention.attention.run_with_device_consumer(
                q_cpu,
                k_cpu,
                v_cpu,
                sequence_meta.cu_seqlens,
                sequence_meta.cu_seqlens,
                output_consumer=self.consumer,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats.projection.attention,
            )
        stats.projection.attention_output_seconds += time.perf_counter() - attention_started

        stats.post_attention_roundtrip_bytes_avoided += 2 * hidden_bytes
        stats.blocks += 1
        elapsed = time.perf_counter() - started
        stats.wall_seconds += elapsed
        stats.projection.wall_seconds += elapsed
        return hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        blocks: Iterable[tuple[H3MaterializedProjection, H3BlockOps]],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: H3DiTStats | None = None,
    ) -> torch.Tensor:
        stats = H3DiTStats() if stats is None else stats
        for projection, ops in blocks:
            self.run_block_(
                hidden_host,
                sequence_meta,
                projection,
                ops,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )
        return hidden_host


__all__ = ["H3MaterializedRunner"]
