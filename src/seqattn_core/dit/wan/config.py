from __future__ import annotations

import os
from dataclasses import dataclass

from ..._config_file import (
    ExecutionMode,
    load_model_config,
    validate_model_config,
)


@dataclass(frozen=True)
class WanConfig:
    execution_mode: ExecutionMode = "materialized"
    projection_tile_tokens: int = 2048
    ffn_tile_tokens: int = 2048

    def __post_init__(self) -> None:
        validate_model_config(self, "wan")


def load_wan_config(path: str | os.PathLike[str] | None = None) -> WanConfig:
    return load_model_config(WanConfig, "wan", path)


__all__ = ["WanConfig", "load_wan_config"]
