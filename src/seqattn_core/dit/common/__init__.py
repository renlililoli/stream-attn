"""Internal mechanisms shared by model-specific DiT runners."""

from .attention import (
    MaterializedAttentionBatch,
    MaterializedAttentionExecutor,
    RecomputedAttentionExecutor,
)
from .consumer import AttentionOutputConsumer, AttentionOutputWorkspace
from .contracts import (
    AttentionEpilogue,
    DeviceTileOp,
    LeaseFactory,
    TiledStageOp,
    require_distinct_storage,
    validate_hidden_host,
)
from .masks import cu_seqlens_from_padding_mask, reject_additive_attention_mask
from .tiled import TiledHostStageRunner, TiledStageStats

__all__ = [
    "AttentionEpilogue",
    "AttentionOutputConsumer",
    "AttentionOutputWorkspace",
    "DeviceTileOp",
    "LeaseFactory",
    "MaterializedAttentionBatch",
    "MaterializedAttentionExecutor",
    "RecomputedAttentionExecutor",
    "TiledHostStageRunner",
    "TiledStageOp",
    "TiledStageStats",
    "cu_seqlens_from_padding_mask",
    "reject_additive_attention_mask",
    "require_distinct_storage",
    "validate_hidden_host",
]
