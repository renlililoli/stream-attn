from __future__ import annotations

import os
from dataclasses import dataclass

from ..._config_file import (
    ExecutionMode,
    execution_mode,
    load_config_table,
    positive_int,
    reject_unknown_keys,
)


@dataclass(frozen=True)
class H3Config:
    execution_mode: ExecutionMode = "materialized"
    projection_tile_tokens: int = 4096
    ffn_tile_tokens: int = 4096

    def validate(self) -> None:
        execution_mode(self.__dict__, "minimax_h3")
        positive_int(self.__dict__, "minimax_h3", "projection_tile_tokens", 4096)
        positive_int(self.__dict__, "minimax_h3", "ffn_tile_tokens", 4096)


def load_h3_config(path: str | os.PathLike[str] | None = None) -> H3Config:
    section = load_config_table("minimax_h3", path)
    reject_unknown_keys(
        section,
        "minimax_h3",
        {"execution_mode", "projection_tile_tokens", "ffn_tile_tokens"},
    )
    config = H3Config(
        execution_mode=execution_mode(section, "minimax_h3"),
        projection_tile_tokens=positive_int(section, "minimax_h3", "projection_tile_tokens", 4096),
        ffn_tile_tokens=positive_int(section, "minimax_h3", "ffn_tile_tokens", 4096),
    )
    return config


__all__ = ["H3Config", "load_h3_config"]
