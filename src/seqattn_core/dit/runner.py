from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..projection import ProjectedAttentionRunner
from ..stats import H3DiTStats
from .consumer import H3DeviceOutputConsumer
from .types import (
    H3BlockOps,
    H3ChunkPlan,
    H3SequenceMeta,
    estimate_h3_aux_workspace_bytes,
)
from .workspace import H3BlockWorkspace


class H3DiTRunner:
    """Materialized-QKV MiniMax-H3 block scheduler with a fused GPU consumer."""

    def __init__(
        self,
        projected_attention: ProjectedAttentionRunner,
        *,
        hidden_features: int,
        mlp_chunk_tokens: int,
        num_final_output_buffers: int = 2,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if mlp_chunk_tokens <= 0:
            raise ValueError("mlp_chunk_tokens must be positive")
        if num_final_output_buffers not in {1, 2}:
            raise ValueError("num_final_output_buffers must be 1 or 2")
        if projected_attention.attention.backend != "triton":
            raise ValueError("the H3 fused block runner requires the Triton backend")
        if projected_attention.plan.output_mode != "device_consumer":
            raise ValueError("the H3 fused block runner requires device_consumer output mode")

        self.projected_attention = projected_attention
        self.hidden_features = hidden_features
        self.mlp_chunk_tokens = mlp_chunk_tokens
        self.num_final_output_buffers = num_final_output_buffers
        attention_plan = projected_attention.plan
        pipeline_config = projected_attention.pipeline_config
        aux_workspace = estimate_h3_aux_workspace_bytes(
            hidden_features=hidden_features,
            dtype=attention_plan.dtype,
            projection_chunk_tokens=pipeline_config.projection_chunk_tokens,
            num_projection_buffers=pipeline_config.num_projection_buffers,
            mlp_chunk_tokens=mlp_chunk_tokens,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.plan = H3ChunkPlan(
            projection_chunk_tokens=pipeline_config.projection_chunk_tokens,
            q_chunk_tokens=attention_plan.q_chunk_tokens,
            kv_chunk_tokens=attention_plan.kv_chunk_tokens,
            mlp_chunk_tokens=mlp_chunk_tokens,
            estimated_workspace_bytes=attention_plan.estimated_workspace_bytes + aux_workspace,
        )
        self.plan.validate()
        self.workspace = H3BlockWorkspace(
            hidden_features=hidden_features,
            mlp_chunk_tokens=mlp_chunk_tokens,
            dtype=attention_plan.dtype,
            device=attention_plan.device,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.consumer = H3DeviceOutputConsumer(self.workspace)

    def _validate_hidden(self, hidden_host: torch.Tensor) -> None:
        plan = self.projected_attention.plan
        if hidden_host.device.type != "cpu" or hidden_host.ndim != 2:
            raise ValueError("hidden_host must use CPU [tokens, hidden_features] layout")
        if hidden_host.shape[1] != self.hidden_features:
            raise ValueError("hidden_host feature size does not match the H3 runner")
        if hidden_host.shape[0] > plan.max_q_tokens:
            raise ValueError("hidden_host token count exceeds the H3 runner plan")
        if hidden_host.dtype != plan.dtype:
            raise ValueError("hidden_host dtype does not match the H3 runner plan")
        if not hidden_host.is_contiguous():
            raise ValueError("hidden_host must be contiguous")
        if (
            self.projected_attention.pipeline_config.require_pinned_hidden
            and not hidden_host.is_pinned()
        ):
            raise ValueError("asynchronous H3 execution requires pinned hidden_host")

    @torch.inference_mode()
    def run_block_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
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
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        started = time.perf_counter()

        with ops.qkv_context():
            q_cpu, k_cpu, v_cpu = self.projected_attention.project_qkv_to_host(
                hidden_host,
                ops.project_qkv,
                stats=stats.projection,
            )
        qkv_host_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
        )
        stats.qkv_host_bytes_peak = max(stats.qkv_host_bytes_peak, qkv_host_bytes)
        raw_attention_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.projection.raw_attention_roundtrip_bytes_avoided += 2 * raw_attention_bytes

        self.consumer.reset(hidden_host=hidden_host, ops=ops, stats=stats)
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

        hidden_bytes = hidden_host.numel() * hidden_host.element_size()
        stats.post_attention_roundtrip_bytes_avoided += 2 * hidden_bytes
        stats.blocks += 1
        elapsed = time.perf_counter() - started
        stats.wall_seconds += elapsed
        stats.projection.wall_seconds += elapsed
        return hidden_host

    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        block_ops: Iterable[H3BlockOps],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: H3DiTStats | None = None,
    ) -> torch.Tensor:
        stats = H3DiTStats() if stats is None else stats
        for ops in block_ops:
            self.run_block_(
                hidden_host,
                sequence_meta,
                ops,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )
        return hidden_host


__all__ = ["H3DiTRunner"]
