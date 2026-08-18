from __future__ import annotations

import torch

from ..kernels import triton_is_available


def resolve_backend(name: str, dtype: torch.dtype, device: torch.device) -> str:
    if name == "auto":
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            return "triton" if triton_is_available() else "reference"
        return "reference"
    if name == "triton":
        if device.type != "cuda":
            raise ValueError("the Triton backend requires a CUDA device")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("the Triton backend requires float16 or bfloat16 inputs")
        if not triton_is_available():
            raise RuntimeError("the Triton backend is not available")
    return name


__all__ = ["resolve_backend"]
