from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingAttentionConfig:
    """Execution policy for CPU-backed exact attention.

    ``workspace_budget_bytes`` covers operator-owned CUDA buffers only.  It is
    intentionally distinct from a whole-process VRAM limit, because CUDA
    context, model weights, and allocations owned by the caller remain outside
    this package.
    """

    workspace_budget_bytes: int | None = None
    q_chunk_tokens: int | None = None
    kv_chunk_tokens: int = 4096
    num_kv_buffers: int = 2
    num_output_buffers: int = 1
    block_m: int = 64
    block_n: int = 64
    num_warps: int = 4
    num_stages: int = 2
    backend: str = "auto"
    require_pinned: bool = True
    pin_output: bool = True
    enable_nvtx: bool = False

    def validate(self) -> None:
        if self.workspace_budget_bytes is not None and self.workspace_budget_bytes <= 0:
            raise ValueError("workspace_budget_bytes must be positive")
        if self.q_chunk_tokens is not None and self.q_chunk_tokens <= 0:
            raise ValueError("q_chunk_tokens must be positive")
        if self.kv_chunk_tokens <= 0:
            raise ValueError("kv_chunk_tokens must be positive")
        if self.num_kv_buffers not in {1, 2, 3}:
            raise ValueError("num_kv_buffers must be 1, 2, or 3")
        if self.num_output_buffers not in {1, 2}:
            raise ValueError("num_output_buffers must be 1 or 2")
        if self.block_m not in {16, 32, 64, 128}:
            raise ValueError("block_m must be one of 16, 32, 64, 128")
        if self.block_n not in {16, 32, 64, 128}:
            raise ValueError("block_n must be one of 16, 32, 64, 128")
        if self.num_warps not in {2, 4, 8}:
            raise ValueError("num_warps must be 2, 4, or 8")
        if self.num_stages not in {1, 2, 3, 4}:
            raise ValueError("num_stages must be within [1, 4]")
        if self.backend not in {"auto", "triton", "reference"}:
            raise ValueError(f"unsupported backend: {self.backend}")


@dataclass(frozen=True)
class ProjectionPipelineConfig:
    """Execution policy for CPU-hidden -> QKV -> attention -> output projection.

    Exact self-attention has a global K/V readiness barrier.  Projection chunks
    are pipelined internally, then attention and output projection run as a
    second pipeline without materializing raw attention output on the CPU.
    """

    projection_chunk_tokens: int = 2048
    num_projection_buffers: int = 2
    require_pinned_hidden: bool = True
    pin_qkv: bool = True
    pin_output: bool = True
    enable_nvtx: bool = False

    def validate(self) -> None:
        if self.projection_chunk_tokens <= 0:
            raise ValueError("projection_chunk_tokens must be positive")
        if self.num_projection_buffers not in {1, 2, 3}:
            raise ValueError("num_projection_buffers must be 1, 2, or 3")
