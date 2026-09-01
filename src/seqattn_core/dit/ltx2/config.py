from __future__ import annotations

import os
from dataclasses import dataclass

from ..._config_file import (
    ExecutionMode,
    load_model_config,
    validate_model_config,
)


@dataclass(frozen=True)
class LTX2Config:
    execution_mode: ExecutionMode = "materialized"
    projection_tile_tokens: int = 2048
    video_ffn_tile_tokens: int = 2048
    audio_ffn_tile_tokens: int = 2048

    def __post_init__(self) -> None:
        validate_model_config(self, "ltx2")


def load_ltx2_config(path: str | os.PathLike[str] | None = None) -> LTX2Config:
    return load_model_config(LTX2Config, "ltx2", path)


__all__ = ["LTX2Config", "load_ltx2_config"]
