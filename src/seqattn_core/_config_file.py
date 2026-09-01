from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Literal, TypeVar, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ExecutionMode = Literal["materialized", "recompute"]
ConfigT = TypeVar("ConfigT")


def default_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "seqattn" / "config.toml"


def load_config_table(
    table_name: str,
    path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    configured_path = os.environ.get("SEQATTN_CONFIG") if path is None else None
    config_path = (
        Path(path).expanduser()
        if path is not None
        else Path(configured_path).expanduser()
        if configured_path
        else default_config_path()
    )
    if not config_path.exists():
        if path is not None or configured_path:
            raise FileNotFoundError(f"SeqAttn config does not exist: {config_path}")
        return {}

    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    section = document.get(table_name, {})
    if not isinstance(section, dict):
        raise TypeError(f"seqattn config [{table_name}] must be a TOML table")
    return section


def positive_int(
    section: Mapping[str, object],
    table_name: str,
    key: str,
    default: int,
) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{table_name}.{key} must be a positive integer")
    return value


def execution_mode(
    section: Mapping[str, object],
    table_name: str,
    default: ExecutionMode = "materialized",
) -> ExecutionMode:
    value = section.get("execution_mode", default)
    if not isinstance(value, str) or value not in {"materialized", "recompute"}:
        raise ValueError(f"{table_name}.execution_mode must be 'materialized' or 'recompute'")
    return cast(ExecutionMode, value)


def reject_unknown_keys(
    section: Mapping[str, object],
    table_name: str,
    allowed: set[str],
) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"seqattn config [{table_name}] has unknown keys: {keys}")


def validate_model_config(config: object, table_name: str) -> None:
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "execution_mode":
            execution_mode({field.name: value}, table_name)
        else:
            positive_int({field.name: value}, table_name, field.name, field.default)


def load_model_config(
    config_type: type[ConfigT],
    table_name: str,
    path: str | os.PathLike[str] | None = None,
) -> ConfigT:
    section = load_config_table(table_name, path)
    config_fields = fields(config_type)
    allowed = {field.name for field in config_fields}
    reject_unknown_keys(section, table_name, allowed)
    values: dict[str, object] = {}
    for field in config_fields:
        if field.name == "execution_mode":
            values[field.name] = execution_mode(section, table_name, field.default)
        else:
            values[field.name] = positive_int(
                section,
                table_name,
                field.name,
                field.default,
            )
    return config_type(**values)


__all__ = [
    "ExecutionMode",
    "default_config_path",
    "execution_mode",
    "load_config_table",
    "load_model_config",
    "positive_int",
    "reject_unknown_keys",
    "validate_model_config",
]
