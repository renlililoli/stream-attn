from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Callable

import torch

from .config import StreamingAttentionConfig
from .kernels import finalize_attention, triton_is_available, update_attention_state
from .planner import AttentionPlan
from .reference import streaming_attention_reference
from .stats import StreamingAttentionStats
from .validation import require_pinned_inputs, validate_host_qkv


def resolve_backend(name: str, dtype: torch.dtype, device: torch.device) -> str:
    if name == "auto":
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            return "triton" if triton_is_available() else "reference"
        return "reference"
    if name == "triton":
        if device.type != "cuda":
            raise ValueError("the Triton backend requires a CUDA device")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("the Triton backend requires float16 or bfloat16 inputs")
        if not triton_is_available():
            raise RuntimeError("the Triton backend is not available")
    return name


class _CudaWorkspace:
    def __init__(self, plan: AttentionPlan):
        device = plan.device
        self.q = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device=device,
        )
        self.k = [
            torch.empty(
                (plan.kv_chunk_tokens, plan.kv_heads, plan.head_dim),
                dtype=plan.dtype,
                device=device,
            )
            for _ in range(plan.num_kv_buffers)
        ]
        self.v = [torch.empty_like(tensor) for tensor in self.k]
        self.running_max = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads), dtype=torch.float32, device=device
        )
        self.running_sum = torch.empty_like(self.running_max)
        self.accumulator = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.output = (
            [torch.empty_like(self.q) for _ in range(plan.num_output_buffers)]
            if plan.output_mode == "host"
            else []
        )
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.q_ready = torch.cuda.Event()
        self.q_free = torch.cuda.Event()
        self.q_has_pending_compute = False
        self.kv_ready = [torch.cuda.Event() for _ in self.k]
        self.kv_free = [torch.cuda.Event() for _ in self.k]
        self.kv_has_pending_compute = [False for _ in self.k]
        self.output_ready = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_free = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_has_pending_copy = [False for _ in range(plan.num_output_buffers)]


class StreamingAttentionRunner:
    """Reusable execution plan and CUDA workspace.

    Reusing a runner avoids allocator churn and keeps stream/event creation out
    of the hot path.  One runner is intentionally single-flight; callers that
    need concurrent independent requests should create one runner per request
    stream.
    """

    def __init__(
        self,
        plan: AttentionPlan,
        config: StreamingAttentionConfig | None = None,
    ):
        self.plan = plan
        self.config = StreamingAttentionConfig() if config is None else config
        self.config.validate()
        if plan.output_mode != self.config.output_mode:
            raise ValueError("attention plan output_mode does not match runner config")
        self.backend = resolve_backend(self.config.backend, plan.dtype, plan.device)
        self._workspace = _CudaWorkspace(plan) if self.backend == "triton" else None

    @torch.inference_mode()
    def __call__(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        stats: StreamingAttentionStats | None = None,
    ) -> torch.Tensor:
        q_bounds, k_bounds = validate_host_qkv(
            q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k
        )
        if self.plan.output_mode != "host":
            raise ValueError(
                "a device_consumer runner requires run_with_device_output()"
            )
        if q_cpu.shape[1:] != (self.plan.q_heads, self.plan.head_dim):
            raise ValueError("q shape does not match the runner plan")
        if k_cpu.shape[1:] != (self.plan.kv_heads, self.plan.head_dim):
            raise ValueError("k/v shape does not match the runner plan")
        if q_cpu.dtype != self.plan.dtype:
            raise ValueError("input dtype does not match the runner plan")
        if q_cpu.shape[0] > self.plan.max_q_tokens or k_cpu.shape[0] > self.plan.max_kv_tokens:
            raise ValueError("input token count exceeds the runner plan")
        scale = self.plan.head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        if out is None:
            out = torch.empty(
                q_cpu.shape,
                dtype=q_cpu.dtype,
                device="cpu",
                pin_memory=self.config.pin_output and torch.cuda.is_available(),
            )
        if out.shape != q_cpu.shape or out.dtype != q_cpu.dtype or out.device.type != "cpu":
            raise ValueError("out must be a CPU tensor matching q shape and dtype")

        if stats is None:
            stats = StreamingAttentionStats()
        stats.backend = self.backend
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        stats.q_chunk_tokens = self.plan.q_chunk_tokens
        stats.kv_chunk_tokens = self.plan.kv_chunk_tokens
        started = time.perf_counter()
        if self.backend == "reference":
            result = streaming_attention_reference(
                q_cpu,
                k_cpu,
                v_cpu,
                cu_seqlens_q,
                cu_seqlens_k,
                q_chunk_tokens=self.plan.q_chunk_tokens,
                kv_chunk_tokens=self.plan.kv_chunk_tokens,
                device=self.plan.device,
                softmax_scale=scale,
                causal=causal,
                out=out,
            )
        else:
            if self.config.require_pinned:
                require_pinned_inputs(q_cpu, k_cpu, v_cpu)
            if self.config.pin_output and not out.is_pinned():
                raise ValueError("asynchronous D2H requires a pinned out tensor")
            result = self._run_triton(
                q_cpu,
                k_cpu,
                v_cpu,
                q_bounds,
                k_bounds,
                scale,
                causal,
                out,
                stats,
            )
        stats.wall_seconds += time.perf_counter() - started
        return result

    @torch.inference_mode()
    def run_with_device_output(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        output_transform: Callable[[torch.Tensor, int, int], torch.Tensor],
        out: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: StreamingAttentionStats | None = None,
    ) -> torch.Tensor:
        """Run attention and consume each GPU output tile before D2H.

        ``output_transform`` receives flattened ``[tokens, q_heads * head_dim]``
        attention output plus global ``start``/``stop`` token offsets.  It must
        return a CUDA tensor matching ``out[start:stop]``.  This hook is intended
        for output projection and its inference epilogue, avoiding a raw
        attention D2H followed by an immediate H2D.
        """

        if self.backend != "triton":
            raise ValueError("device output transforms require the Triton backend")
        q_bounds, k_bounds = validate_host_qkv(
            q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k
        )
        if q_cpu.shape[1:] != (self.plan.q_heads, self.plan.head_dim):
            raise ValueError("q shape does not match the runner plan")
        if k_cpu.shape[1:] != (self.plan.kv_heads, self.plan.head_dim):
            raise ValueError("k/v shape does not match the runner plan")
        if q_cpu.dtype != self.plan.dtype:
            raise ValueError("input dtype does not match the runner plan")
        if q_cpu.shape[0] > self.plan.max_q_tokens or k_cpu.shape[0] > self.plan.max_kv_tokens:
            raise ValueError("input token count exceeds the runner plan")
        if out.device.type != "cpu" or out.shape[0] != q_cpu.shape[0]:
            raise ValueError("out must be a CPU tensor with one row per query token")
        if self.config.require_pinned:
            require_pinned_inputs(q_cpu, k_cpu, v_cpu)
        if self.config.pin_output and not out.is_pinned():
            raise ValueError("asynchronous D2H requires a pinned out tensor")

        scale = self.plan.head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        if stats is None:
            stats = StreamingAttentionStats()
        stats.backend = self.backend
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        stats.q_chunk_tokens = self.plan.q_chunk_tokens
        stats.kv_chunk_tokens = self.plan.kv_chunk_tokens
        started = time.perf_counter()
        result = self._run_triton(
            q_cpu,
            k_cpu,
            v_cpu,
            q_bounds,
            k_bounds,
            scale,
            causal,
            out,
            stats,
            output_transform=output_transform,
        )
        stats.wall_seconds += time.perf_counter() - started
        return result

    def _range(self, name: str):
        if self.config.enable_nvtx:
            return torch.cuda.nvtx.range(name)
        return nullcontext()

    def _run_triton(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        q_bounds: list[int],
        k_bounds: list[int],
        scale: float,
        causal: bool,
        out_cpu: torch.Tensor,
        stats: StreamingAttentionStats,
        output_transform: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        workspace = self._workspace
        assert workspace is not None
        plan = self.plan
        compute_stream = workspace.compute_stream
        q_chunk_index = 0

        with torch.cuda.device(plan.device):
            for q_start, q_stop, k_start, k_stop in zip(
                q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]
            ):
                q_length = q_stop - q_start
                k_length = k_stop - k_start
                causal_shift = k_length - q_length
                for q_tile_start in range(q_start, q_stop, plan.q_chunk_tokens):
                    q_tile_stop = min(q_tile_start + plan.q_chunk_tokens, q_stop)
                    q_tokens = q_tile_stop - q_tile_start
                    q_local_offset = q_tile_start - q_start
                    output_index = q_chunk_index % plan.num_output_buffers
                    reuse_q_for_output = (
                        output_transform is not None
                        and plan.output_mode == "device_consumer"
                    )

                    with self._range("seqattn:q_h2d"):
                        with torch.cuda.stream(workspace.h2d_stream):
                            if workspace.q_has_pending_compute:
                                workspace.h2d_stream.wait_event(workspace.q_free)
                            workspace.q[:q_tokens].copy_(
                                q_cpu[q_tile_start:q_tile_stop],
                                non_blocking=q_cpu.is_pinned(),
                            )
                            workspace.q_ready.record(workspace.h2d_stream)
                    stats.h2d_bytes += (
                        q_tokens
                        * q_cpu.shape[1]
                        * q_cpu.shape[2]
                        * q_cpu.element_size()
                    )
                    stats.q_chunks += 1
                    stats.max_resident_q_tokens = max(stats.max_resident_q_tokens, q_tokens)
                    with torch.cuda.stream(compute_stream):
                        compute_stream.wait_event(workspace.q_ready)

                        initialize = True
                        for kv_tile_index, kv_tile_start in enumerate(
                            range(k_start, k_stop, plan.kv_chunk_tokens)
                        ):
                            kv_tile_stop = min(
                                kv_tile_start + plan.kv_chunk_tokens, k_stop
                            )
                            kv_tokens = kv_tile_stop - kv_tile_start
                            buffer_index = kv_tile_index % plan.num_kv_buffers
                            with self._range("seqattn:kv_h2d"):
                                with torch.cuda.stream(workspace.h2d_stream):
                                    if workspace.kv_has_pending_compute[buffer_index]:
                                        workspace.h2d_stream.wait_event(
                                            workspace.kv_free[buffer_index]
                                        )
                                    workspace.k[buffer_index][:kv_tokens].copy_(
                                        k_cpu[kv_tile_start:kv_tile_stop],
                                        non_blocking=k_cpu.is_pinned(),
                                    )
                                    workspace.v[buffer_index][:kv_tokens].copy_(
                                        v_cpu[kv_tile_start:kv_tile_stop],
                                        non_blocking=v_cpu.is_pinned(),
                                    )
                                    workspace.kv_ready[buffer_index].record(
                                        workspace.h2d_stream
                                    )
                            stats.h2d_bytes += (
                                2
                                * kv_tokens
                                * k_cpu.shape[1]
                                * k_cpu.shape[2]
                                * k_cpu.element_size()
                            )
                            stats.kv_tiles += 1
                            compute_stream.wait_event(
                                workspace.kv_ready[buffer_index]
                            )
                            with self._range("seqattn:fused_update"):
                                update_attention_state(
                                    workspace.q,
                                    workspace.k[buffer_index],
                                    workspace.v[buffer_index],
                                    workspace.running_max,
                                    workspace.running_sum,
                                    workspace.accumulator,
                                    q_tokens=q_tokens,
                                    kv_tokens=kv_tokens,
                                    q_local_offset=q_local_offset,
                                    kv_local_offset=kv_tile_start - k_start,
                                    causal_shift=causal_shift,
                                    softmax_scale=scale,
                                    causal=causal,
                                    initialize=initialize,
                                    block_m=plan.block_m,
                                    block_n=plan.block_n,
                                    num_warps=plan.num_warps,
                                    num_stages=plan.num_stages,
                                )
                            initialize = False
                            workspace.kv_free[buffer_index].record(compute_stream)
                            workspace.kv_has_pending_compute[buffer_index] = True

                        if not reuse_q_for_output:
                            workspace.q_free.record(compute_stream)
                            workspace.q_has_pending_compute = True
                        else:
                            workspace.q_has_pending_compute = False
                        if (
                            not reuse_q_for_output
                            and workspace.output_has_pending_copy[output_index]
                        ):
                            compute_stream.wait_event(workspace.output_free[output_index])
                        finalize_output = (
                            workspace.q
                            if reuse_q_for_output
                            else workspace.output[output_index]
                        )
                        with self._range("seqattn:fused_finalize"):
                            finalize_attention(
                                workspace.accumulator,
                                workspace.running_sum,
                                finalize_output,
                                q_tokens=q_tokens,
                            )
                        output_gpu = finalize_output[:q_tokens]
                        output_aliases_q = reuse_q_for_output
                        if output_transform is not None:
                            with self._range("seqattn:device_output_transform"):
                                output_gpu = output_transform(
                                    output_gpu.reshape(q_tokens, -1),
                                    q_tile_start,
                                    q_tile_stop,
                                )
                            if output_gpu.device.type != "cuda":
                                raise ValueError("output_transform must return a CUDA tensor")
                            output_slice_shape = out_cpu[q_tile_start:q_tile_stop].shape
                            if output_gpu.shape != output_slice_shape:
                                raise ValueError(
                                    "output_transform result shape does not match "
                                    "the output slice: "
                                    f"{tuple(output_gpu.shape)} != "
                                    f"{tuple(output_slice_shape)}"
                                )
                            if output_gpu.dtype != out_cpu.dtype:
                                raise ValueError(
                                    "output_transform result dtype must match out dtype"
                                )
                            output_aliases_q = (
                                output_gpu.untyped_storage().data_ptr()
                                == workspace.q.untyped_storage().data_ptr()
                            )
                            if reuse_q_for_output and not output_aliases_q:
                                workspace.q_free.record(compute_stream)
                                workspace.q_has_pending_compute = True
                        workspace.output_ready[output_index].record(compute_stream)
                    with self._range("seqattn:output_d2h"):
                        with torch.cuda.stream(workspace.d2h_stream):
                            workspace.d2h_stream.wait_event(
                                workspace.output_ready[output_index]
                            )
                            out_cpu[q_tile_start:q_tile_stop].copy_(
                                output_gpu, non_blocking=out_cpu.is_pinned()
                            )
                            output_gpu.record_stream(workspace.d2h_stream)
                            workspace.output_free[output_index].record(
                                workspace.d2h_stream
                            )
                            if reuse_q_for_output and output_aliases_q:
                                workspace.q_free.record(workspace.d2h_stream)
                                workspace.q_has_pending_compute = True
                    workspace.output_has_pending_copy[output_index] = True
                    stats.d2h_bytes += output_gpu.numel() * output_gpu.element_size()
                    q_chunk_index += 1
            workspace.d2h_stream.synchronize()
        return out_cpu
