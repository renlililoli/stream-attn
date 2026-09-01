from __future__ import annotations

from dataclasses import dataclass

from ...config import ProjectionPipelineConfig, StreamingAttentionConfig
from ...planner import AttentionPlan
from ...projection import (
    MaterializedQKVArena,
    ProjectedAttentionRunner,
    ProjectedCrossAttentionRunner,
    RecomputedAttentionRunner,
    RecomputedCrossAttentionRunner,
)
from .config import LTX2Config, load_ltx2_config
from .materialized_runner import LTX2MaterializedRunner
from .recompute_runner import LTX2RecomputeRunner


@dataclass(frozen=True)
class LTX2AttentionPlans:
    video_self_attention: AttentionPlan
    audio_self_attention: AttentionPlan
    video_text_attention: AttentionPlan
    audio_text_attention: AttentionPlan
    video_from_audio_attention: AttentionPlan
    audio_from_video_attention: AttentionPlan


LTX2Runner = LTX2MaterializedRunner | LTX2RecomputeRunner


def build_ltx2_runner(
    plans: LTX2AttentionPlans,
    *,
    video_hidden_features: int,
    audio_hidden_features: int,
    text_hidden_features: int,
    config: LTX2Config | None = None,
    attention_config: StreamingAttentionConfig | None = None,
    num_output_buffers: int = 2,
) -> LTX2Runner:
    """Construct the configured single-GPU LTX2 runner."""

    config = load_ltx2_config() if config is None else config
    if config.execution_mode == "materialized":
        pipeline = ProjectionPipelineConfig(
            projection_tile_tokens=config.projection_tile_tokens,
        )
        video_arena = MaterializedQKVArena.for_plans(
            (
                plans.video_self_attention,
                plans.video_text_attention,
                plans.video_from_audio_attention,
            ),
            pin_memory=pipeline.pin_qkv,
        )
        audio_arena = MaterializedQKVArena.for_plans(
            (
                plans.audio_self_attention,
                plans.audio_text_attention,
                plans.audio_from_video_attention,
            ),
            pin_memory=pipeline.pin_qkv,
        )
        return LTX2MaterializedRunner(
            video_self_attention=ProjectedAttentionRunner(
                plans.video_self_attention,
                attention_config,
                pipeline,
                arena=video_arena,
            ),
            audio_self_attention=ProjectedAttentionRunner(
                plans.audio_self_attention,
                attention_config,
                pipeline,
                arena=audio_arena,
            ),
            video_text_attention=ProjectedCrossAttentionRunner(
                plans.video_text_attention,
                attention_config,
                pipeline,
                arena=video_arena,
            ),
            audio_text_attention=ProjectedCrossAttentionRunner(
                plans.audio_text_attention,
                attention_config,
                pipeline,
                arena=audio_arena,
            ),
            video_from_audio_attention=ProjectedCrossAttentionRunner(
                plans.video_from_audio_attention,
                attention_config,
                pipeline,
                arena=video_arena,
            ),
            audio_from_video_attention=ProjectedCrossAttentionRunner(
                plans.audio_from_video_attention,
                attention_config,
                pipeline,
                arena=audio_arena,
            ),
            video_hidden_features=video_hidden_features,
            audio_hidden_features=audio_hidden_features,
            video_ffn_tile_tokens=config.video_ffn_tile_tokens,
            audio_ffn_tile_tokens=config.audio_ffn_tile_tokens,
            num_output_buffers=num_output_buffers,
        )
    return LTX2RecomputeRunner(
        video_self_attention=RecomputedAttentionRunner(
            plans.video_self_attention,
            hidden_features=video_hidden_features,
            attention_config=attention_config,
        ),
        audio_self_attention=RecomputedAttentionRunner(
            plans.audio_self_attention,
            hidden_features=audio_hidden_features,
            attention_config=attention_config,
        ),
        video_text_attention=RecomputedCrossAttentionRunner(
            plans.video_text_attention,
            query_hidden_features=video_hidden_features,
            context_hidden_features=text_hidden_features,
            attention_config=attention_config,
        ),
        audio_text_attention=RecomputedCrossAttentionRunner(
            plans.audio_text_attention,
            query_hidden_features=audio_hidden_features,
            context_hidden_features=text_hidden_features,
            attention_config=attention_config,
        ),
        video_from_audio_attention=RecomputedCrossAttentionRunner(
            plans.video_from_audio_attention,
            query_hidden_features=video_hidden_features,
            context_hidden_features=audio_hidden_features,
            attention_config=attention_config,
        ),
        audio_from_video_attention=RecomputedCrossAttentionRunner(
            plans.audio_from_video_attention,
            query_hidden_features=audio_hidden_features,
            context_hidden_features=video_hidden_features,
            attention_config=attention_config,
        ),
        video_ffn_tile_tokens=config.video_ffn_tile_tokens,
        audio_ffn_tile_tokens=config.audio_ffn_tile_tokens,
        num_output_buffers=num_output_buffers,
    )


__all__ = ["LTX2AttentionPlans", "LTX2Runner", "build_ltx2_runner"]
