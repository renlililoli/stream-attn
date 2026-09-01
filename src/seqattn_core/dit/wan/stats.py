from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...stats import (
    ProjectedAttentionStats,
    ProjectedCrossAttentionStats,
    RecomputedAttentionStats,
    RecomputedCrossAttentionStats,
)
from ..common import TiledStageStats


@dataclass
class WanDiTStats:
    backend: str = ""
    qkv_storage_policy: str = ""
    wall_seconds: float = 0.0
    blocks: int = 0
    hidden_host_bytes_peak: int = 0
    qkv_host_bytes_peak: int = 0
    self_attention: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    cross_attention: ProjectedCrossAttentionStats = field(
        default_factory=ProjectedCrossAttentionStats
    )
    self_recompute: RecomputedAttentionStats = field(default_factory=RecomputedAttentionStats)
    cross_recompute: RecomputedCrossAttentionStats = field(
        default_factory=RecomputedCrossAttentionStats
    )
    ffn: TiledStageStats = field(default_factory=TiledStageStats)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["WanDiTStats"]
