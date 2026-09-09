from __future__ import annotations

from dataclasses import replace

from ...config import ProjectionPipelineConfig
from ...plan import AttentionPlan
from ...projection import ProjectedAttentionRunner, RecomputedAttentionRunner
from ...sparse import SolStreamingAttentionRunner, build_sol_streaming_plan
from ...streaming import StreamingAttentionRunner
from .config import H3Config, load_h3_config
from .materialized_runner import H3MaterializedRunner
from .recompute_runner import H3RecomputeRunner

H3Runner = H3MaterializedRunner | H3RecomputeRunner


def build_h3_runner(
    plan: AttentionPlan,
    *,
    hidden_features: int,
    config: H3Config | None = None,
    num_output_buffers: int = 2,
) -> H3Runner:
    """Construct the configured single-GPU H3 runner from one resolved plan."""

    config = load_h3_config() if config is None else config
    sol_plan = None
    sol_attention = None
    if config.attention_mode == "sol_streaming":
        sol_base = replace(
            plan,
            backend="triton",
            estimated_workspace_bytes=plan.estimated_workspace_bytes - plan.backend_workspace_bytes,
            backend_workspace_bytes=0,
        )
        sol_plan = build_sol_streaming_plan(sol_base)
        sol_attention = SolStreamingAttentionRunner(
            sol_plan,
            StreamingAttentionRunner(sol_plan.attention),
        )
    if config.execution_mode == "materialized":
        projected = ProjectedAttentionRunner(
            plan,
            ProjectionPipelineConfig(
                projection_tile_tokens=config.projection_tile_tokens,
            ),
        )
        return H3MaterializedRunner(
            projected,
            hidden_features=hidden_features,
            ffn_tile_tokens=config.ffn_tile_tokens,
            num_final_output_buffers=num_output_buffers,
            config=config,
            sol_attention=sol_attention,
        )
    recomputed = RecomputedAttentionRunner(
        plan,
        hidden_features=hidden_features,
    )
    return H3RecomputeRunner(
        recomputed,
        ffn_tile_tokens=config.ffn_tile_tokens,
        num_final_output_buffers=num_output_buffers,
        config=config,
        sol_attention=sol_attention,
    )


__all__ = ["H3Runner", "build_h3_runner"]
