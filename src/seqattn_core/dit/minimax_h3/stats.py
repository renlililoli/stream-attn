from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...stats import ProjectedAttentionStats, RecomputedAttentionStats


@dataclass
class H3DiTStats:
    backend: str = ""
    qkv_storage_policy: str = ""
    wall_seconds: float = 0.0
    blocks: int = 0
    ffn_tiles: int = 0
    ffn_cross_q_boundaries: int = 0
    final_hidden_d2h_bytes: int = 0
    post_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes_peak: int = 0
    hidden_host_bytes_peak: int = 0
    estimated_workspace_bytes: int = 0
    projection: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    recompute: RecomputedAttentionStats = field(default_factory=RecomputedAttentionStats)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["H3DiTStats"]
