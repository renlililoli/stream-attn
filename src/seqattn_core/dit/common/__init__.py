"""Internal mechanisms shared by model-specific DiT runners."""

from .consumer import AttentionOutputConsumer, AttentionOutputWorkspace
from .masks import cu_seqlens_from_padding_mask, reject_additive_attention_mask
from .tiled import TiledHostStageRunner, TiledStageStats
from .types import AttentionEpilogue, DeviceTileOp, LeaseFactory, TiledStageOp
from .validation import require_distinct_storage, validate_hidden_host

__all__ = [
    "AttentionEpilogue",
    "AttentionOutputConsumer",
    "AttentionOutputWorkspace",
    "DeviceTileOp",
    "LeaseFactory",
    "TiledHostStageRunner",
    "TiledStageOp",
    "TiledStageStats",
    "cu_seqlens_from_padding_mask",
    "reject_additive_attention_mask",
    "require_distinct_storage",
    "validate_hidden_host",
]
