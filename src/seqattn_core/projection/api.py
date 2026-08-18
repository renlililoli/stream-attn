from __future__ import annotations

import torch

from ..config import ProjectionPipelineConfig, StreamingAttentionConfig
from ..planner import build_plan
from ..stats import ProjectedAttentionStats
from .runner import ProjectedAttentionRunner
from .types import OutputProjector, QKVProjector


def streaming_projected_self_attention(
    hidden_cpu: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    project_qkv: QKVProjector,
    output_projector: OutputProjector,
    output_features: int,
    attention_config: StreamingAttentionConfig | None = None,
    pipeline_config: ProjectionPipelineConfig | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    stats: ProjectedAttentionStats | None = None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Functional convenience wrapper for one projected self-attention call."""

    attention_config = StreamingAttentionConfig() if attention_config is None else attention_config
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=hidden_cpu.dtype,
        device=device,
        max_q_tokens=hidden_cpu.shape[0],
        max_kv_tokens=hidden_cpu.shape[0],
        config=attention_config,
    )
    runner = ProjectedAttentionRunner(plan, attention_config, pipeline_config)
    return runner(
        hidden_cpu,
        cu_seqlens,
        project_qkv=project_qkv,
        output_projector=output_projector,
        output_features=output_features,
        softmax_scale=softmax_scale,
        causal=causal,
        stats=stats,
    )


__all__ = ["streaming_projected_self_attention"]
