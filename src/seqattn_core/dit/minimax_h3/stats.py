from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ...sparse import SolStreamingStats
from ...stats import ProjectedAttentionStats, RecomputedAttentionStats


@dataclass
class H3DiTStats:
    backend: str = ""
    qkv_storage_policy: str = ""
    wall_seconds: float = 0.0
    blocks: int = 0
    dense_attention_blocks: int = 0
    sol_streaming_blocks: int = 0
    ffn_tiles: int = 0
    ffn_cross_q_boundaries: int = 0
    final_hidden_d2h_bytes: int = 0
    post_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes_peak: int = 0
    hidden_host_bytes_peak: int = 0
    estimated_workspace_bytes: int = 0
    projection: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    recompute: RecomputedAttentionStats = field(default_factory=RecomputedAttentionStats)
    sol_attention: SolStreamingStats = field(default_factory=SolStreamingStats)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["sol_attention"] = self.sol_attention.as_dict()
        return result


__all__ = ["H3DiTStats"]
