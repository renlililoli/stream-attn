from __future__ import annotations

import os
from collections.abc import Collection

import torch

from .._config_file import load_config_table
from ..kernels import triton_is_available
from .flash_backends import flash_backend_is_available

_ALIASES = {
    "builtin": "triton",
    "flash2": "fa2",
    "flash2_split": "fa2",
}
_CUDA_BACKENDS = {"triton", "fa2", "fa3", "fa4"}
_FLASH_BACKENDS = {"fa2", "fa3", "fa4"}
_KNOWN_BACKENDS = {"auto", "reference", *_CUDA_BACKENDS, *_ALIASES}


def canonical_backend_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in _KNOWN_BACKENDS:
        raise ValueError(f"unsupported backend: {name}")
    return _ALIASES.get(normalized, normalized)


def _backend_from_config_file() -> str | None:
    section = load_config_table("attention")
    backend = section.get("backend")
    if backend is None:
        return None
    if not isinstance(backend, str):
        raise ValueError("seqattn config attention.backend must be a string")  # noqa: TRY004
    return canonical_backend_name(backend)


def configured_backend_name(explicit: str | None) -> str:
    if explicit is not None:
        return canonical_backend_name(explicit)
    environment = os.environ.get("SEQATTN_BACKEND")
    if environment:
        return canonical_backend_name(environment)
    return _backend_from_config_file() or "auto"


def backend_is_available(name: str) -> bool:
    name = canonical_backend_name(name)
    if name == "reference":
        return True
    if name == "triton":
        return triton_is_available()
    return flash_backend_is_available(name) if name in _FLASH_BACKENDS else False


def automatic_backend_order(device: torch.device) -> tuple[str, ...]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ("reference",)
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 12:
        return ("triton", "fa4", "reference")
    if major == 10:
        return ("fa4", "triton", "reference")
    if major == 9:
        return ("fa3", "fa2", "triton", "reference")
    if major == 8:
        return ("fa2", "triton", "reference")
    return ("triton", "reference")


def _validate_backend_capability(
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
    head_dim: int | None,
) -> None:
    if backend not in _CUDA_BACKENDS:
        return
    if device.type != "cuda":
        raise ValueError(f"the {backend} backend requires a CUDA device")
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"the {backend} backend requires float16 or bfloat16 inputs")
    if head_dim is not None:
        if backend == "triton" and head_dim < 16:
            raise ValueError(
                "the builtin backend requires head_dim >= 16 (tl.dot needs BLOCK_D >= 16)"
            )
        if backend in _FLASH_BACKENDS and (head_dim % 8 or head_dim > 256):
            raise ValueError(f"the {backend} backend requires head_dim divisible by 8 and <= 256")
    if backend in {"fa3", "fa4"} and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        if backend == "fa3" and major != 9:
            raise ValueError("the fa3 backend requires an SM90 GPU")
        if backend == "fa4" and major < 10:
            raise ValueError("the fa4 backend requires a Blackwell GPU")


def resolve_backend(
    name: str | None,
    dtype: torch.dtype,
    device: torch.device,
    *,
    head_dim: int | None = None,
    allowed: Collection[str] | None = None,
) -> str:
    requested = configured_backend_name(name)
    allowed_canonical = (
        None if allowed is None else {canonical_backend_name(candidate) for candidate in allowed}
    )
    if requested == "auto":
        if dtype not in {torch.float16, torch.bfloat16}:
            candidates = ("reference",)
        else:
            candidates = automatic_backend_order(device)
        for backend in candidates:
            if allowed_canonical is not None and backend not in allowed_canonical:
                continue
            if backend_is_available(backend):
                _validate_backend_capability(backend, dtype, device, head_dim)
                return backend
        raise RuntimeError("no compatible seqattn backend is available")

    backend = requested
    if allowed_canonical is not None and backend not in allowed_canonical:
        choices = ", ".join(sorted(allowed_canonical))
        raise ValueError(f"backend {backend!r} is not supported by this runtime; choose {choices}")
    _validate_backend_capability(backend, dtype, device, head_dim)
    if not backend_is_available(backend):
        package = {"fa2": "flash-attn", "fa3": "FlashAttention-3", "fa4": "flash-attn-4"}
        requirement = package.get(backend, backend)
        raise RuntimeError(f"the {backend} backend requires {requirement}")
    return backend


__all__ = [
    "automatic_backend_order",
    "backend_is_available",
    "canonical_backend_name",
    "configured_backend_name",
    "resolve_backend",
]
