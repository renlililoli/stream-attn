from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class H3TileConfig:
    qkv_tile_tokens: int = 2048
    mlp_tile_tokens: int = 2048

    def validate(self) -> None:
        for name, value in (
            ("qkv_tile_tokens", self.qkv_tile_tokens),
            ("mlp_tile_tokens", self.mlp_tile_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"minimax_h3.{name} must be a positive integer")


def _default_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "seqattn" / "config.toml"


def load_h3_tile_config(path: str | os.PathLike[str] | None = None) -> H3TileConfig:
    configured_path = os.environ.get("SEQATTN_CONFIG") if path is None else None
    config_path = (
        Path(path).expanduser()
        if path is not None
        else Path(configured_path).expanduser()
        if configured_path
        else _default_config_path()
    )
    if not config_path.exists():
        if path is not None or configured_path:
            raise FileNotFoundError(f"SEQATTN_CONFIG does not exist: {config_path}")
        return H3TileConfig()

    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    section = document.get("minimax_h3", {})
    if not isinstance(section, dict):
        raise TypeError("seqattn config [minimax_h3] must be a TOML table")

    config = H3TileConfig(
        qkv_tile_tokens=section.get("qkv_tile_tokens", 2048),
        mlp_tile_tokens=section.get("mlp_tile_tokens", 2048),
    )
    config.validate()
    return config


__all__ = ["H3TileConfig", "load_h3_tile_config"]
