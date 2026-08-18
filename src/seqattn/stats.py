from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StreamingAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    q_chunks: int = 0
    kv_tiles: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    estimated_workspace_bytes: int = 0
    q_chunk_tokens: int = 0
    kv_chunk_tokens: int = 0
    max_resident_q_tokens: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass
class ProjectedAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    projection_seconds: float = 0.0
    attention_output_seconds: float = 0.0
    projection_chunks: int = 0
    projection_hidden_h2d_bytes: int = 0
    projection_qkv_d2h_bytes: int = 0
    raw_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes: int = 0
    attention: StreamingAttentionStats = field(default_factory=StreamingAttentionStats)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["total_h2d_bytes"] = (
            self.projection_hidden_h2d_bytes + self.attention.h2d_bytes
        )
        result["total_d2h_bytes"] = (
            self.projection_qkv_d2h_bytes + self.attention.d2h_bytes
        )
        return result
