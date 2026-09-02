from __future__ import annotations

from ...config import ProjectionPipelineConfig
from ...plan import AttentionPlan
from ...projection import ProjectedAttentionRunner, RecomputedAttentionRunner
from ...sparse import SolStreamingAttentionRunner, build_sol_streaming_plan
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
    sol_plan = build_sol_streaming_plan(plan) if config.attention_mode == "sol_streaming" else None
    runtime_plan = plan if sol_plan is None else sol_plan.attention
    if config.execution_mode == "materialized":
        projected = ProjectedAttentionRunner(
            runtime_plan,
            ProjectionPipelineConfig(
                projection_tile_tokens=config.projection_tile_tokens,
            ),
        )
        sol_attention = (
            None if sol_plan is None else SolStreamingAttentionRunner(sol_plan, projected.attention)
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
        runtime_plan,
        hidden_features=hidden_features,
    )
    sol_attention = (
        None if sol_plan is None else SolStreamingAttentionRunner(sol_plan, recomputed.attention)
    )
    return H3RecomputeRunner(
        recomputed,
        ffn_tile_tokens=config.ffn_tile_tokens,
        num_final_output_buffers=num_output_buffers,
        config=config,
        sol_attention=sol_attention,
    )


__all__ = ["H3Runner", "build_h3_runner"]
