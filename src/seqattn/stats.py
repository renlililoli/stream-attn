from __future__ import annotations

from dataclasses import asdict, dataclass


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
