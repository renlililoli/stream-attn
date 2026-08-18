"""Compatibility facade for the projected attention pipeline."""

from seqattn_core.projection import (
    OutputProjector,
    ProjectedAttentionRunner,
    QKVProjector,
    streaming_projected_self_attention,
)

__all__ = [
    "OutputProjector",
    "ProjectedAttentionRunner",
    "QKVProjector",
    "streaming_projected_self_attention",
]
