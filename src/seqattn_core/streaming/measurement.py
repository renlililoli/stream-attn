from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryTaskMeasurement:
    h2d_seconds: float = 0.0
    attention_seconds: float = 0.0
    consumer_seconds: float = 0.0
    d2h_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    attention_flops: int = 0


__all__ = ["QueryTaskMeasurement"]
