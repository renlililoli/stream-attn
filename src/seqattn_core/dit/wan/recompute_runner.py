from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ...projection import RecomputedAttentionRunner, RecomputedCrossAttentionRunner
from ..common import (
    AttentionOutputConsumer,
    AttentionOutputWorkspace,
    TiledHostStageRunner,
    require_distinct_storage,
    validate_hidden_host,
)
from .stats import WanDiTStats
from .types import WanBlockOps, WanRecomputeProjections, WanSequenceMeta


class WanRecomputeRunner:
    """Wan block runner without sequence-sized host Q/K/V tensors."""

    def __init__(
        self,
        self_attention: RecomputedAttentionRunner,
        cross_attention: RecomputedCrossAttentionRunner,
        *,
        ffn_chunk_tokens: int,
        num_output_buffers: int = 2,
    ) -> None:
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.hidden_features = self_attention.hidden_features
        self._validate_plans()
        plan = self_attention.plan
        output_chunk_tokens = max(plan.q_chunk_tokens, cross_attention.plan.q_chunk_tokens)
        self.output_workspace = AttentionOutputWorkspace(
            hidden_features=self.hidden_features,
            output_chunk_tokens=output_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
            num_output_buffers=num_output_buffers,
        )
        self.consumer = AttentionOutputConsumer(self.output_workspace)
        self.ffn = TiledHostStageRunner(
            hidden_features=self.hidden_features,
            chunk_tokens=ffn_chunk_tokens,
            dtype=plan.dtype,
            device=plan.device,
            require_pinned_hidden=self_attention.require_pinned_hidden,
        )

    def _validate_plans(self) -> None:
        self_plan = self.self_attention.plan
        cross_plan = self.cross_attention.plan
        if self_plan.device != cross_plan.device or self_plan.dtype != cross_plan.dtype:
            raise ValueError("Wan self and cross attention must use the same device and dtype")
        if self_plan.max_q_tokens != cross_plan.max_q_tokens:
            raise ValueError("Wan self and cross attention must plan the same hidden token count")
        if self_plan.max_q_tokens != self_plan.max_kv_tokens:
            raise ValueError("Wan self-attention requires equal planned Q and K/V token counts")
        if self.cross_attention.query_hidden_features != self.hidden_features:
            raise ValueError("Wan cross-attention query feature size must match hidden features")

    def _validate_inputs(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
    ) -> None:
        for name, tensor in (
            ("source_hidden_host", source_hidden_host),
            ("destination_hidden_host", destination_hidden_host),
        ):
            validate_hidden_host(
                tensor,
                plan=self.self_attention.plan,
                hidden_features=self.hidden_features,
                require_pinned=self.self_attention.require_pinned_hidden,
                name=name,
            )
        require_distinct_storage(source_hidden_host, destination_hidden_host)
        self.cross_attention._validate_hidden(
            text_hidden_host,
            hidden_features=self.cross_attention.context_hidden_features,
            max_tokens=self.cross_attention.plan.max_kv_tokens,
            name="text_hidden_host",
        )
        sequence_meta.validate(source_hidden_host.shape[0], text_hidden_host.shape[0])

    @torch.inference_mode()
    def run_block(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        projections: WanRecomputeProjections,
        ops: WanBlockOps,
        *,
        self_softmax_scale: float | None = None,
        cross_softmax_scale: float | None = None,
        self_causal: bool = False,
        stats: WanDiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            source_hidden_host,
            destination_hidden_host,
            text_hidden_host,
            sequence_meta,
        )
        stats = WanDiTStats() if stats is None else stats
        stats.backend = self.self_attention.attention.backend
        stats.qkv_storage_policy = "recompute"
        started = time.perf_counter()

        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=source_hidden_host,
            epilogue=ops.self_attention_epilogue,
        )
        with projections.self_attention.context(), ops.self_attention_context():
            self.self_attention.run_with_device_consumer(
                source_hidden_host,
                sequence_meta.hidden_cu_seqlens,
                project_q=projections.self_attention.project_q,
                project_kv=projections.self_attention.project_kv,
                output_consumer=self.consumer,
                softmax_scale=self_softmax_scale,
                causal=self_causal,
                stats=stats.self_recompute,
            )

        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=destination_hidden_host,
            epilogue=ops.cross_attention_epilogue,
        )
        with projections.text_cross_attention.context(), ops.cross_attention_context():
            self.cross_attention.run_with_device_consumer(
                destination_hidden_host,
                text_hidden_host,
                sequence_meta.hidden_cu_seqlens,
                sequence_meta.text_cu_seqlens,
                project_q=projections.text_cross_attention.project_q,
                project_kv=projections.text_cross_attention.project_kv,
                output_consumer=self.consumer,
                softmax_scale=cross_softmax_scale,
                stats=stats.cross_recompute,
            )

        with ops.ffn_context():
            self.ffn.run(
                destination_hidden_host,
                destination_hidden_host,
                ops.ffn,
                stats=stats.ffn,
            )
        hidden_bytes = source_hidden_host.numel() * source_hidden_host.element_size()
        stats.hidden_host_bytes_peak = max(stats.hidden_host_bytes_peak, 2 * hidden_bytes)
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return destination_hidden_host

    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        scratch_hidden_host: torch.Tensor,
        text_hidden_host: torch.Tensor,
        sequence_meta: WanSequenceMeta,
        blocks: Iterable[tuple[WanRecomputeProjections, WanBlockOps]],
        **kwargs,
    ) -> torch.Tensor:
        stats = kwargs.pop("stats", None)
        stats = WanDiTStats() if stats is None else stats
        source = hidden_host
        destination = scratch_hidden_host
        for projections, ops in blocks:
            self.run_block(
                source,
                destination,
                text_hidden_host,
                sequence_meta,
                projections,
                ops,
                stats=stats,
                **kwargs,
            )
            source, destination = destination, source
        return source


__all__ = ["WanRecomputeRunner"]
