from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import ProjectedAttentionRunner, ProjectedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    TiledHostStageRunner,
    validate_hidden_host,
)
from .stats import WanDiTStats
from .types import WanBlockOps, WanMaterializedProjections, WanSequenceMeta


class WanMaterializedRunner:
    """Wan block order: self-attention, text cross-attention, then FFN."""

    def __init__(
        self,
        self_attention: ProjectedAttentionRunner,
        cross_attention: ProjectedCrossAttentionRunner,
        *,
        hidden_features: int,
        ffn_tile_tokens: int,
        num_output_buffers: int = 2,
    ) -> None:
        init_single_flight(self)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.hidden_features = hidden_features
        self._validate_plans()
        plan = self_attention.plan
        output_chunk_tokens = max(plan.q_chunk_tokens, cross_attention.plan.q_chunk_tokens)
        self.output_workspace = AttentionOutputWorkspace(
            hidden_features=hidden_features,
            output_chunk_tokens=output_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
            num_output_buffers=num_output_buffers,
        )
        self.consumer = AttentionOutputConsumer(self.output_workspace)
        self.ffn = TiledHostStageRunner(
            hidden_features=hidden_features,
            chunk_tokens=ffn_tile_tokens,
            dtype=plan.dtype,
            device=plan.device,
            require_pinned_hidden=self_attention.pipeline_config.require_pinned_hidden,
        )

    def _validate_plans(self) -> None:
        self_plan = self.self_attention.plan
        cross_plan = self.cross_attention.plan
        if self.hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if (
            self_plan.output_mode != "device_consumer"
            or cross_plan.output_mode != "device_consumer"
        ):
            raise ValueError("Wan attention runners require device_consumer output mode")
        if self_plan.device != cross_plan.device or self_plan.dtype != cross_plan.dtype:
            raise ValueError("Wan self and cross attention must use the same device and dtype")
        if self_plan.max_q_tokens != cross_plan.max_q_tokens:
            raise ValueError("Wan self and cross attention must plan the same hidden token count")
        if self_plan.max_q_tokens != self_plan.max_kv_tokens:
            raise ValueError("Wan self-attention requires equal planned Q and K/V token counts")

    def _validate_inputs(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
    ) -> None:
        validate_hidden_host(
            hidden_host,
            plan=self.self_attention.plan,
            hidden_features=self.hidden_features,
            require_pinned=self.self_attention.pipeline_config.require_pinned_hidden,
        )
        self.cross_attention.validate_inputs(hidden_host, text_hidden_host)
        sequence_meta.validate(hidden_host.shape[0], text_hidden_host.shape[0])

    @single_flight
    @torch.inference_mode()
    def run_block_(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        projections: WanMaterializedProjections,
        ops: WanBlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_host, text_hidden_host, sequence_meta)
        stats = WanDiTStats() if stats is None else stats
        stats.backend = self.self_attention.attention.backend
        stats.qkv_storage_policy = "materialized"
        started = time.perf_counter()

        self_stage_started = time.perf_counter()
        with projections.self_attention.context():
            q, k, v = self.self_attention.project_qkv_to_host(
                hidden_host,
                projections.self_attention.project_qkv,
                stats=stats.self_attention,
            )
        raw_output_bytes = q.numel() * q.element_size()
        stats.self_attention.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        self.consumer.reset(
            destination_hidden_host=hidden_host,
            residual_hidden_host=hidden_host,
            epilogue=ops.self_attention_epilogue,
        )
        self_attention_started = time.perf_counter()
        with ops.self_attention_context():
            self.self_attention.attention.run_with_device_consumer(
                q,
                k,
                v,
                sequence_meta.hidden_cu_seqlens,
                sequence_meta.hidden_cu_seqlens,
                output_consumer=self.consumer,
                softmax_scale=self_softmax_scale,
                causal=self_causal,
                stats=stats.self_attention.attention,
            )
        stats.self_attention.attention_output_seconds += (
            time.perf_counter() - self_attention_started
        )
        stats.self_attention.wall_seconds += time.perf_counter() - self_stage_started

        cross_stage_started = time.perf_counter()

        with projections.text_cross_attention.context():
            q, k, v = self.cross_attention.project_to_host(
                hidden_host,
                text_hidden_host,
                project_q=projections.text_cross_attention.project_q,
                project_kv=projections.text_cross_attention.project_kv,
                stats=stats.cross_attention,
            )
        raw_output_bytes = q.numel() * q.element_size()
        stats.cross_attention.raw_attention_roundtrip_bytes_avoided += 2 * raw_output_bytes
        arenas = {
            id(runner.arena): runner.arena for runner in (self.self_attention, self.cross_attention)
        }
        stats.qkv_host_bytes_peak = max(
            stats.qkv_host_bytes_peak,
            sum(arena.allocated_bytes for arena in arenas.values()),
        )
        self.consumer.reset(
            destination_hidden_host=hidden_host,
            residual_hidden_host=hidden_host,
            epilogue=ops.cross_attention_epilogue,
        )
        cross_attention_started = time.perf_counter()
        with ops.cross_attention_context():
            self.cross_attention.attention.run_with_device_consumer(
                q,
                k,
                v,
                sequence_meta.hidden_cu_seqlens,
                sequence_meta.text_cu_seqlens,
                output_consumer=self.consumer,
                softmax_scale=cross_softmax_scale,
                stats=stats.cross_attention.attention,
            )
        stats.cross_attention.attention_output_seconds += (
            time.perf_counter() - cross_attention_started
        )
        stats.cross_attention.wall_seconds += time.perf_counter() - cross_stage_started

        with ops.ffn_context():
            self.ffn.run(hidden_host, hidden_host, ops.ffn, stats=stats.ffn)
        stats.hidden_host_bytes_peak = max(
            stats.hidden_host_bytes_peak,
            (hidden_host.numel() + text_hidden_host.numel()) * hidden_host.element_size(),
        )
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        blocks: Iterable[tuple[WanMaterializedProjections, WanBlockOps]],
        **kwargs,
    ) -> torch.Tensor:
        stats = kwargs.pop("stats", None)
        stats = WanDiTStats() if stats is None else stats
        for projections, ops in blocks:
            self.run_block_(
                hidden_host,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                stats=stats,
                **kwargs,
            )
        return hidden_host


__all__ = ["WanMaterializedRunner"]
