from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, cast

from ..._config_file import (
    ExecutionMode,
    execution_mode,
    load_config_table,
    positive_int,
    reject_unknown_keys,
)
from .types import H3DenoisingStep, H3SequenceMeta

H3AttentionMode = Literal["dense", "sol_streaming"]


@dataclass(frozen=True)
class H3Config:
    execution_mode: ExecutionMode = "materialized"
    attention_mode: H3AttentionMode = "dense"
    projection_tile_tokens: int = 4096
    ffn_tile_tokens: int = 4096
    sol_tau: float = 1.0
    sol_first_dense_step_fraction: float = 0.2
    sol_first_dense_layers: int = 2

    def __post_init__(self) -> None:
        execution_mode({"execution_mode": self.execution_mode}, "minimax_h3")
        if self.attention_mode not in {"dense", "sol_streaming"}:
            raise ValueError("minimax_h3.attention_mode must be 'dense' or 'sol_streaming'")
        for name, value in (
            ("projection_tile_tokens", self.projection_tile_tokens),
            ("ffn_tile_tokens", self.ffn_tile_tokens),
        ):
            positive_int({name: value}, "minimax_h3", name, value)
        if isinstance(self.sol_tau, bool) or not isinstance(self.sol_tau, (int, float)):
            raise TypeError("minimax_h3.sol_tau must be a finite number")
        if not math.isfinite(float(self.sol_tau)):
            raise ValueError("minimax_h3.sol_tau must be a finite number")
        fraction = self.sol_first_dense_step_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise TypeError("minimax_h3.sol_first_dense_step_fraction must be within [0, 1]")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("minimax_h3.sol_first_dense_step_fraction must be within [0, 1]")
        layers = self.sol_first_dense_layers
        if isinstance(layers, bool) or not isinstance(layers, int) or layers < 0:
            raise ValueError("minimax_h3.sol_first_dense_layers must be non-negative")


def use_sol_streaming(
    config: H3Config,
    *,
    sequence_meta: H3SequenceMeta,
    denoising_step: H3DenoisingStep | None,
    block_index: int | None,
) -> bool:
    """Return the explicit H3 dense/sparse policy decision for one DiT block."""

    if config.attention_mode == "dense":
        return False
    if denoising_step is None:
        raise ValueError("sol_streaming requires explicit H3 denoising step metadata")
    denoising_step.validate()
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("sol_streaming requires a non-negative H3 block_index")
    if sequence_meta.exact_prefix_tokens is None:
        raise ValueError("sol_streaming requires exact_prefix_tokens for every packed segment")
    dense_steps = math.ceil(denoising_step.total_steps * config.sol_first_dense_step_fraction)
    return denoising_step.step_index >= dense_steps and block_index >= config.sol_first_dense_layers


def load_h3_config(path: str | os.PathLike[str] | None = None) -> H3Config:
    section = load_config_table("minimax_h3", path)
    allowed = {
        "execution_mode",
        "attention_mode",
        "projection_tile_tokens",
        "ffn_tile_tokens",
        "sol_tau",
        "sol_first_dense_step_fraction",
        "sol_first_dense_layers",
    }
    reject_unknown_keys(section, "minimax_h3", allowed)
    defaults = H3Config()
    attention_mode = section.get("attention_mode", defaults.attention_mode)
    if not isinstance(attention_mode, str):
        raise TypeError("minimax_h3.attention_mode must be 'dense' or 'sol_streaming'")
    return H3Config(
        execution_mode=execution_mode(section, "minimax_h3", defaults.execution_mode),
        attention_mode=cast(H3AttentionMode, attention_mode),
        projection_tile_tokens=positive_int(
            section,
            "minimax_h3",
            "projection_tile_tokens",
            defaults.projection_tile_tokens,
        ),
        ffn_tile_tokens=positive_int(
            section,
            "minimax_h3",
            "ffn_tile_tokens",
            defaults.ffn_tile_tokens,
        ),
        sol_tau=_number(section, "sol_tau", defaults.sol_tau),
        sol_first_dense_step_fraction=_number(
            section,
            "sol_first_dense_step_fraction",
            defaults.sol_first_dense_step_fraction,
        ),
        sol_first_dense_layers=_non_negative_int(
            section,
            "sol_first_dense_layers",
            defaults.sol_first_dense_layers,
        ),
    )


def _number(section: dict[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"minimax_h3.{key} must be a number")
    return float(value)


def _non_negative_int(section: dict[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"minimax_h3.{key} must be a non-negative integer")
    return value


__all__ = ["H3AttentionMode", "H3Config", "load_h3_config", "use_sol_streaming"]
