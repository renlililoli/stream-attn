from __future__ import annotations

import torch

from ..kernels import initialize_split_attention_state, merge_split_attention_state
from ..stats import StreamingAttentionStats
from .flash_backends import flash_partial_forward


class FlashSplitExecutorMixin:
    def _run_flash_split(
        self,
        backend: str,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        q_bounds: list[int],
        k_bounds: list[int],
        scale: float,
        causal: bool,
        out_cpu: torch.Tensor,
        stats: StreamingAttentionStats,
    ) -> torch.Tensor:
        if causal:
            raise ValueError(
                f"{backend} does not support external causal offsets; use backend='builtin'"
            )

        workspace = self._workspace
        assert workspace is not None
        plan = self.plan
        compute_stream = workspace.compute_stream
        q_chunk_index = 0

        with torch.cuda.device(plan.device):
            workspace.pipeline_start.record(compute_stream)
            for q_start, q_stop, k_start, k_stop in zip(
                q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
            ):
                for q_tile_start in range(q_start, q_stop, plan.q_chunk_tokens):
                    q_tile_stop = min(q_tile_start + plan.q_chunk_tokens, q_stop)
                    q_tokens = q_tile_stop - q_tile_start
                    output_index = q_chunk_index % plan.num_output_buffers

                    with torch.cuda.stream(workspace.h2d_stream):
                        if workspace.q_has_pending_compute:
                            workspace.h2d_stream.wait_event(workspace.q_free)
                        workspace.q[:q_tokens].copy_(
                            q_cpu[q_tile_start:q_tile_stop],
                            non_blocking=q_cpu.is_pinned(),
                        )
                        workspace.q_ready.record(workspace.h2d_stream)
                    stats.h2d_bytes += (
                        q_tokens * q_cpu.shape[1] * q_cpu.shape[2] * q_cpu.element_size()
                    )
                    stats.q_chunks += 1
                    stats.max_resident_q_tokens = max(stats.max_resident_q_tokens, q_tokens)

                    with torch.cuda.stream(compute_stream):
                        compute_stream.wait_event(workspace.q_ready)
                        if workspace.output_has_pending_copy[output_index]:
                            compute_stream.wait_event(workspace.output_free[output_index])
                        partial_buffer = workspace.output[output_index][:q_tokens].unsqueeze(0)
                        state_output = workspace.accumulator[:q_tokens].unsqueeze(0)
                        state_lse = workspace.running_max[:q_tokens].unsqueeze(0)
                        initialize = True

                        for kv_tile_index, kv_tile_start in enumerate(
                            range(k_start, k_stop, plan.kv_chunk_tokens)
                        ):
                            kv_tile_stop = min(kv_tile_start + plan.kv_chunk_tokens, k_stop)
                            kv_tokens = kv_tile_stop - kv_tile_start
                            buffer_index = kv_tile_index % plan.num_kv_buffers
                            with torch.cuda.stream(workspace.h2d_stream):
                                if workspace.kv_has_pending_compute[buffer_index]:
                                    workspace.h2d_stream.wait_event(workspace.kv_free[buffer_index])
                                workspace.k[buffer_index][:kv_tokens].copy_(
                                    k_cpu[kv_tile_start:kv_tile_stop],
                                    non_blocking=k_cpu.is_pinned(),
                                )
                                workspace.v[buffer_index][:kv_tokens].copy_(
                                    v_cpu[kv_tile_start:kv_tile_stop],
                                    non_blocking=v_cpu.is_pinned(),
                                )
                                workspace.kv_ready[buffer_index].record(workspace.h2d_stream)
                            stats.h2d_bytes += (
                                2
                                * kv_tokens
                                * k_cpu.shape[1]
                                * k_cpu.shape[2]
                                * k_cpu.element_size()
                            )
                            stats.kv_tiles += 1

                            compute_stream.wait_event(workspace.kv_ready[buffer_index])
                            partial_output, partial_lse = flash_partial_forward(
                                backend,
                                workspace.q[:q_tokens].unsqueeze(0),
                                workspace.k[buffer_index][:kv_tokens].unsqueeze(0),
                                workspace.v[buffer_index][:kv_tokens].unsqueeze(0),
                                partial_buffer,
                                softmax_scale=scale,
                            )
                            if initialize:
                                initialize_split_attention_state(
                                    partial_output, partial_lse, state_output, state_lse
                                )
                                initialize = False
                            else:
                                merge_split_attention_state(
                                    partial_output, partial_lse, state_output, state_lse
                                )
                            workspace.kv_free[buffer_index].record(compute_stream)
                            workspace.kv_has_pending_compute[buffer_index] = True

                        workspace.q_free.record(compute_stream)
                        workspace.q_has_pending_compute = True
                        partial_buffer.copy_(state_output)
                        workspace.output_ready[output_index].record(compute_stream)

                    with torch.cuda.stream(workspace.d2h_stream):
                        workspace.d2h_stream.wait_event(workspace.output_ready[output_index])
                        output_gpu = workspace.output[output_index][:q_tokens]
                        out_cpu[q_tile_start:q_tile_stop].copy_(
                            output_gpu, non_blocking=out_cpu.is_pinned()
                        )
                        workspace.output_free[output_index].record(workspace.d2h_stream)
                    workspace.output_has_pending_copy[output_index] = True
                    stats.d2h_bytes += output_gpu.numel() * output_gpu.element_size()
                    q_chunk_index += 1

            workspace.pipeline_end.record(compute_stream)
            workspace.d2h_stream.synchronize()
            workspace.pipeline_end.synchronize()
            stats.compute_pipeline_seconds += (
                workspace.pipeline_start.elapsed_time(workspace.pipeline_end) / 1000.0
            )
        return out_cpu


__all__ = ["FlashSplitExecutorMixin"]
