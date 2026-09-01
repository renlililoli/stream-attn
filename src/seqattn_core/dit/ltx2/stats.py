from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...stats import ProjectedAttentionStats, ProjectedCrossAttentionStats
from ..common import TiledStageStats


@dataclass
class LTX2DiTStats:
    backend: str = ""
    wall_seconds: float = 0.0
    blocks: int = 0
    hidden_host_bytes_peak: int = 0
    qkv_host_bytes_peak: int = 0
    video_self_attention: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    audio_self_attention: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    video_text_attention: ProjectedCrossAttentionStats = field(
        default_factory=ProjectedCrossAttentionStats
    )
    audio_text_attention: ProjectedCrossAttentionStats = field(
        default_factory=ProjectedCrossAttentionStats
    )
    video_from_audio_attention: ProjectedCrossAttentionStats = field(
        default_factory=ProjectedCrossAttentionStats
    )
    audio_from_video_attention: ProjectedCrossAttentionStats = field(
        default_factory=ProjectedCrossAttentionStats
    )
    video_ffn: TiledStageStats = field(default_factory=TiledStageStats)
    audio_ffn: TiledStageStats = field(default_factory=TiledStageStats)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["LTX2DiTStats"]
