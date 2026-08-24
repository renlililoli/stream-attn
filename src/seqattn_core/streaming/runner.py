from __future__ import annotations

import time
from collections.abc import Callable

import torch

from ..config import StreamingAttentionConfig
from ..planner import AttentionPlan
from ..reference import streaming_attention_reference
from ..stats import StreamingAttentionStats
from ..validation import require_pinned_inputs, validate_host_qkv
from .backend import configured_backend_name, resolve_backend
from .executor import TritonExecutorMixin
from .flash_split_executor import FlashSplitExecutorMixin
from .workspace import CudaWorkspace


class StreamingAttentionRunner(TritonExecutorMixin, FlashSplitExecutorMixin):
    """Reusable execution plan and CUDA workspace.

    One runner is intentionally single-flight. Create one runner per request
    stream when independent calls need to execute concurrently.
    """

    def __init__(
        self,
        plan: AttentionPlan,
        config: StreamingAttentionConfig | None = None,
    ) -> None:
        self.plan = plan
        self.config = StreamingAttentionConfig() if config is None else config
        self.config.validate()
        if plan.output_mode != self.config.output_mode:
            raise ValueError("attention plan output_mode does not match runner config")
        self._backend_request = configured_backend_name(self.config.backend)
        allowed = (
            {"triton", "reference"}
            if plan.output_mode == "device_consumer"
            else {"triton", "fa2", "fa3", "fa4", "reference"}
        )
        self.backend = resolve_backend(
            self._backend_request,
            plan.dtype,
            plan.device,
            head_dim=plan.head_dim,
            allowed=allowed,
        )
        self._workspace = (
            CudaWorkspace(plan)
            if self.backend in {"triton", "fa2", "fa3", "fa4"}
            else None
        )

    def _validate_inputs(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        q_bounds, k_bounds = validate_host_qkv(q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k)
        if q_cpu.shape[1:] != (self.plan.q_heads, self.plan.head_dim):
            raise ValueError("q shape does not match the runner plan")
        if k_cpu.shape[1:] != (self.plan.kv_heads, self.plan.head_dim):
            raise ValueError("k/v shape does not match the runner plan")
        if q_cpu.dtype != self.plan.dtype:
            raise ValueError("input dtype does not match the runner plan")
        if q_cpu.shape[0] > self.plan.max_q_tokens or k_cpu.shape[0] > self.plan.max_kv_tokens:
            raise ValueError("input token count exceeds the runner plan")
        return q_bounds, k_bounds

    def _prepare_triton_io(
        self,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self.config.require_pinned:
            require_pinned_inputs(q_cpu, k_cpu, v_cpu)
        if self.config.pin_output and not out.is_pinned():
            raise ValueError("asynchronous D2H requires a pinned out tensor")

    def _prepare_stats(
        self,
        stats: StreamingAttentionStats | None,
        *,
        backend: str | None = None,
    ) -> StreamingAttentionStats:
        stats = StreamingAttentionStats() if stats is None else stats
        stats.backend = self.backend if backend is None else backend
        stats.estimated_workspace_bytes = self.plan.estimated_workspace_bytes
        stats.q_chunk_tokens = self.plan.q_chunk_tokens
        stats.kv_chunk_tokens = self.plan.kv_chunk_tokens
        return stats

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
        q_bounds, k_bounds = self._validate_inputs(q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k)
        if self.plan.output_mode != "host":
            raise ValueError("a device_consumer runner requires run_with_device_output()")
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

        execution_backend = self.backend
        if causal and execution_backend in {"fa2", "fa3", "fa4"}:
            if self._backend_request != "auto":
                raise ValueError(
                    f"{execution_backend} does not support external causal offsets; "
                    "use backend='builtin'"
                )
            execution_backend = resolve_backend(
                "builtin",
                self.plan.dtype,
                self.plan.device,
                head_dim=self.plan.head_dim,
            )

        stats = self._prepare_stats(stats, backend=execution_backend)
        started = time.perf_counter()
        if execution_backend == "reference":
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
        elif execution_backend == "triton":
            self._prepare_triton_io(q_cpu, k_cpu, v_cpu, out)
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
        else:
            self._prepare_triton_io(q_cpu, k_cpu, v_cpu, out)
            result = self._run_flash_split(
                execution_backend,
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
        """Consume each GPU output tile before D2H."""

        if self.backend != "triton":
            raise ValueError("device output transforms require the Triton backend")
        q_bounds, k_bounds = self._validate_inputs(q_cpu, k_cpu, v_cpu, cu_seqlens_q, cu_seqlens_k)
        if out.device.type != "cpu" or out.shape[0] != q_cpu.shape[0]:
            raise ValueError("out must be a CPU tensor with one row per query token")
        self._prepare_triton_io(q_cpu, k_cpu, v_cpu, out)

        scale = self.plan.head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        stats = self._prepare_stats(stats)
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


__all__ = ["StreamingAttentionRunner"]
