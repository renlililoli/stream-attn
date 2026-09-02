from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from ..._single_flight import init_single_flight, single_flight
from ...projection import RecomputedAttentionRunner
from ...sparse import SolStreamingAttentionRunner
from ..common import require_distinct_storage, validate_hidden_host
from .config import H3Config, use_sol_streaming
from .consumer import H3DeviceOutputConsumer
from .stats import H3DiTStats
from .types import (
    H3BlockOps,
    H3DenoisingStep,
    H3RecomputePlan,
    H3RecomputeProjection,
    H3SequenceMeta,
    estimate_h3_recompute_aux_workspace_bytes,
)
from .workspace import H3BlockWorkspace


class H3RecomputeRunner:
    """MiniMax-H3 scheduler recomputing large Q and K/V tiles from source hidden."""

    def __init__(
        self,
        recomputed_attention: RecomputedAttentionRunner,
        *,
        ffn_tile_tokens: int,
        num_final_output_buffers: int = 2,
        config: H3Config | None = None,
        sol_attention: SolStreamingAttentionRunner | None = None,
    ) -> None:
        init_single_flight(self)
        if ffn_tile_tokens <= 0:
            raise ValueError("ffn_tile_tokens must be positive")
        if num_final_output_buffers not in {1, 2}:
            raise ValueError("num_final_output_buffers must be 1 or 2")
        if recomputed_attention.attention.backend != "triton":
            raise ValueError("the H3 recompute runner requires the Triton backend")
        config = H3Config(execution_mode="recompute") if config is None else config
        if config.execution_mode != "recompute":
            raise ValueError("H3 recompute runner requires execution_mode='recompute'")
        if config.attention_mode == "sol_streaming" and sol_attention is None:
            raise ValueError("H3 sol_streaming mode requires a Sol attention runner")
        if sol_attention is not None and sol_attention.plan.attention != recomputed_attention.plan:
            raise ValueError("H3 recompute and Sol attention plans must match")

        self.recomputed_attention = recomputed_attention
        self.config = config
        self.sol_attention = sol_attention
        self.hidden_features = recomputed_attention.hidden_features
        self.ffn_tile_tokens = ffn_tile_tokens
        attention_plan = recomputed_attention.plan
        hidden_staging_tokens = recomputed_attention.workspace.staging_tokens
        aux_workspace = estimate_h3_recompute_aux_workspace_bytes(
            hidden_features=self.hidden_features,
            dtype=attention_plan.dtype,
            hidden_staging_tokens=hidden_staging_tokens,
            ffn_tile_tokens=ffn_tile_tokens,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.plan = H3RecomputePlan(
            q_chunk_tokens=attention_plan.q_chunk_tokens,
            kv_chunk_tokens=attention_plan.kv_chunk_tokens,
            ffn_tile_tokens=ffn_tile_tokens,
            hidden_staging_tokens=hidden_staging_tokens,
            estimated_workspace_bytes=attention_plan.estimated_workspace_bytes + aux_workspace,
        )
        self.plan.validate()
        self.workspace = H3BlockWorkspace(
            hidden_features=self.hidden_features,
            ffn_tile_tokens=ffn_tile_tokens,
            dtype=attention_plan.dtype,
            device=attention_plan.device,
            num_final_output_buffers=num_final_output_buffers,
        )
        self.consumer = H3DeviceOutputConsumer(self.workspace)

    def _validate_hidden(self, hidden_host: torch.Tensor, *, name: str) -> None:
        validate_hidden_host(
            hidden_host,
            plan=self.recomputed_attention.plan,
            hidden_features=self.hidden_features,
            require_pinned=self.recomputed_attention.require_pinned_hidden,
            name=name,
        )

    @single_flight
    @torch.inference_mode()
    def run_block(
        self,
        source_hidden_host: torch.Tensor,
        destination_hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        projection: H3RecomputeProjection,
        ops: H3BlockOps,
        *,
        block_index: int | None = None,
        denoising_step: H3DenoisingStep | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: H3DiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_hidden(source_hidden_host, name="source_hidden_host")
        self._validate_hidden(destination_hidden_host, name="destination_hidden_host")
        require_distinct_storage(source_hidden_host, destination_hidden_host)
        sequence_meta.validate(source_hidden_host.shape[0])
        sparse = use_sol_streaming(
            self.config,
            sequence_meta=sequence_meta,
            denoising_step=denoising_step,
            block_index=block_index,
        )
        if sparse and causal:
            raise ValueError("sol_streaming does not support causal attention")

        stats = H3DiTStats() if stats is None else stats
        stats.backend = self.recomputed_attention.attention.backend
        stats.qkv_storage_policy = "recompute"
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        started = time.perf_counter()

        self.consumer.reset(
            destination_hidden_host=destination_hidden_host,
            residual_hidden_host=source_hidden_host,
            ops=ops,
            stats=stats,
        )
        with projection.context(), ops.consumer_context():
            if sparse:
                assert self.sol_attention is not None
                assert sequence_meta.exact_prefix_tokens is not None

                def execute(source) -> None:
                    self.sol_attention.run_with_qkv_source(
                        source,
                        source_hidden_host.shape[0],
                        sequence_meta.cu_seqlens,
                        exact_prefix_tokens=sequence_meta.exact_prefix_tokens,
                        output_consumer=self.consumer,
                        tau=self.config.sol_tau,
                        softmax_scale=softmax_scale,
                        stats=stats.sol_attention,
                    )

                self.recomputed_attention.run_with_source_executor(
                    source_hidden_host,
                    project_q=projection.project_q,
                    project_kv=projection.project_kv,
                    execute=execute,
                    stats=stats.recompute,
                    range_name="seqattn:h3_sol_recomputed_attention",
                )
                stats.sol_streaming_blocks += 1
            else:
                self.recomputed_attention.run_with_device_consumer(
                    source_hidden_host,
                    sequence_meta.cu_seqlens,
                    project_q=projection.project_q,
                    project_kv=projection.project_kv,
                    output_consumer=self.consumer,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    stats=stats.recompute,
                )
                stats.dense_attention_blocks += 1

        hidden_bytes = source_hidden_host.numel() * source_hidden_host.element_size()
        stats.hidden_host_bytes_peak = max(stats.hidden_host_bytes_peak, 2 * hidden_bytes)
        stats.post_attention_roundtrip_bytes_avoided += 2 * hidden_bytes
        stats.recompute.qkv_host_bytes = 0
        stats.blocks += 1
        stats.wall_seconds += time.perf_counter() - started
        return destination_hidden_host

    @single_flight
    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        scratch_hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        blocks: Iterable[tuple[H3RecomputeProjection, H3BlockOps]],
        *,
        first_block_index: int = 0,
        denoising_step: H3DenoisingStep | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: H3DiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_hidden(hidden_host, name="hidden_host")
        self._validate_hidden(scratch_hidden_host, name="scratch_hidden_host")
        require_distinct_storage(hidden_host, scratch_hidden_host)
        stats = H3DiTStats() if stats is None else stats
        source = hidden_host
        destination = scratch_hidden_host
        for block_index, (projection, ops) in enumerate(blocks, start=first_block_index):
            self.run_block(
                source,
                destination,
                sequence_meta,
                projection,
                ops,
                block_index=block_index,
                denoising_step=denoising_step,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )
            source, destination = destination, source
        return source


__all__ = ["H3RecomputeRunner"]
