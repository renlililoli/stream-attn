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
class WanConfig:
    execution_mode: ExecutionMode = "materialized"
    projection_tile_tokens: int = 2048
    ffn_tile_tokens: int = 2048

    def validate(self) -> None:
        execution_mode(self.__dict__, "wan")
        positive_int(self.__dict__, "wan", "projection_tile_tokens", 2048)
        positive_int(self.__dict__, "wan", "ffn_tile_tokens", 2048)


def load_wan_config(path: str | os.PathLike[str] | None = None) -> WanConfig:
    section = load_config_table("wan", path)
    reject_unknown_keys(
        section,
        "wan",
        {"execution_mode", "projection_tile_tokens", "ffn_tile_tokens"},
    )
    return WanConfig(
        execution_mode=execution_mode(section, "wan"),
        projection_tile_tokens=positive_int(section, "wan", "projection_tile_tokens", 2048),
        ffn_tile_tokens=positive_int(section, "wan", "ffn_tile_tokens", 2048),
    )


__all__ = ["WanConfig", "load_wan_config"]
