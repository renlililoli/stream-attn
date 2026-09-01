from __future__ import annotations

import os
from dataclasses import dataclass

from ..._config_file import (
    ExecutionMode,
    load_model_config,
    validate_model_config,
)


@dataclass(frozen=True)
class H3Config:
    execution_mode: ExecutionMode = "materialized"
    projection_tile_tokens: int = 4096
    ffn_tile_tokens: int = 4096

    def __post_init__(self) -> None:
        validate_model_config(self, "minimax_h3")


def load_h3_config(path: str | os.PathLike[str] | None = None) -> H3Config:
    return load_model_config(H3Config, "minimax_h3", path)


__all__ = ["H3Config", "load_h3_config"]
