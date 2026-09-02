from __future__ import annotations

from dataclasses import dataclass

from ...config import ProjectionPipelineConfig
from ...plan import AttentionPlan
from ...projection import (
    MaterializedQKVArena,
    ProjectedAttentionRunner,
    ProjectedCrossAttentionRunner,
    RecomputedAttentionRunner,
    RecomputedCrossAttentionRunner,
)
from .config import WanConfig, load_wan_config
from .materialized_runner import WanMaterializedRunner
from .recompute_runner import WanRecomputeRunner


@dataclass(frozen=True)
class WanAttentionPlans:
    self_attention: AttentionPlan
    text_cross_attention: AttentionPlan


WanRunner = WanMaterializedRunner | WanRecomputeRunner


def build_wan_runner(
    plans: WanAttentionPlans,
    *,
    hidden_features: int,
    text_hidden_features: int,
    config: WanConfig | None = None,
    num_output_buffers: int = 2,
) -> WanRunner:
    """Construct the configured single-GPU Wan runner."""

    config = load_wan_config() if config is None else config
    if config.execution_mode == "materialized":
        pipeline = ProjectionPipelineConfig(
            projection_tile_tokens=config.projection_tile_tokens,
        )
        arena = MaterializedQKVArena.for_plans(
            (plans.self_attention, plans.text_cross_attention),
            pin_memory=pipeline.pin_qkv,
        )
        self_attention = ProjectedAttentionRunner(
            plans.self_attention,
            pipeline,
            arena=arena,
        )
        cross_attention = ProjectedCrossAttentionRunner(
            plans.text_cross_attention,
            pipeline,
            query_hidden_features=hidden_features,
            context_hidden_features=text_hidden_features,
            arena=arena,
        )
        return WanMaterializedRunner(
            self_attention,
            cross_attention,
            hidden_features=hidden_features,
            ffn_tile_tokens=config.ffn_tile_tokens,
            num_output_buffers=num_output_buffers,
        )
    self_attention = RecomputedAttentionRunner(
        plans.self_attention,
        hidden_features=hidden_features,
    )
    cross_attention = RecomputedCrossAttentionRunner(
        plans.text_cross_attention,
        query_hidden_features=hidden_features,
        context_hidden_features=text_hidden_features,
    )
    return WanRecomputeRunner(
        self_attention,
        cross_attention,
        ffn_tile_tokens=config.ffn_tile_tokens,
        num_output_buffers=num_output_buffers,
    )


__all__ = ["WanAttentionPlans", "WanRunner", "build_wan_runner"]
