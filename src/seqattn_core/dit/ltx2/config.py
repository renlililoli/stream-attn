from __future__ import annotations

import os
from dataclasses import dataclass

from ..._config_file import load_config_table, positive_int, reject_unknown_keys


@dataclass(frozen=True)
class LTX2Config:
    projection_tile_tokens: int = 2048
    video_ffn_tile_tokens: int = 2048
    audio_ffn_tile_tokens: int = 2048

    def validate(self) -> None:
        for key in (
            "projection_tile_tokens",
            "video_ffn_tile_tokens",
            "audio_ffn_tile_tokens",
        ):
            positive_int(self.__dict__, "ltx2", key, 2048)


def load_ltx2_config(path: str | os.PathLike[str] | None = None) -> LTX2Config:
    section = load_config_table("ltx2", path)
    if "execution_mode" in section:
        raise ValueError(
            "ltx2.execution_mode is unsupported; LTX2 currently supports materialized execution only"
        )
    reject_unknown_keys(
        section,
        "ltx2",
        {"projection_tile_tokens", "video_ffn_tile_tokens", "audio_ffn_tile_tokens"},
    )
    return LTX2Config(
        projection_tile_tokens=positive_int(section, "ltx2", "projection_tile_tokens", 2048),
        video_ffn_tile_tokens=positive_int(section, "ltx2", "video_ffn_tile_tokens", 2048),
        audio_ffn_tile_tokens=positive_int(section, "ltx2", "audio_ffn_tile_tokens", 2048),
    )


__all__ = ["LTX2Config", "load_ltx2_config"]
