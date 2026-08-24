from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


class FlashPartialBackend(Protocol):
    name: str

    def forward_partial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _normalize_result(
    backend: str,
    result: object,
    output_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)) or len(result) < 2:
        raise RuntimeError(f"{backend} did not return output and FP32 LSE")
    returned_output, lse = result[:2]
    if not isinstance(returned_output, torch.Tensor) or not isinstance(lse, torch.Tensor):
        raise RuntimeError(f"{backend} returned an invalid output/LSE pair")
    if returned_output.shape != output_buffer.shape:
        raise RuntimeError(
            f"{backend} returned output shape {tuple(returned_output.shape)}, "
            f"expected {tuple(output_buffer.shape)}"
        )
    if returned_output.data_ptr() != output_buffer.data_ptr():
        output_buffer.copy_(returned_output)

    batch, tokens, heads, _ = output_buffer.shape
    if lse.shape == (batch, heads, tokens):
        normalized_lse = lse
    elif lse.shape == (batch, tokens, heads):
        normalized_lse = lse.transpose(1, 2)
    else:
        raise RuntimeError(
            f"{backend} returned LSE shape {tuple(lse.shape)}, "
            f"expected {(batch, heads, tokens)}"
        )
    return output_buffer, normalized_lse.contiguous().float()


@dataclass(frozen=True)
class Flash2Backend:
    name: str = "fa2"

    @staticmethod
    def load_extension():
        from flash_attn.flash_attn_interface import flash_attn_gpu

        return flash_attn_gpu

    def forward_partial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extension = self.load_extension()
        result = extension.fwd(
            q,
            k,
            v,
            output,
            None,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            False,
            None,
        )
        return _normalize_result(self.name, result, output)


@dataclass(frozen=True)
class Flash3Backend:
    name: str = "fa3"

    @staticmethod
    def load_function():
        for module_name in (
            "flash_attn_interface",
            "flash_attn_3.flash_attn_interface",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            function = getattr(module, "flash_attn_func", None)
            if function is not None:
                return function
        raise ImportError("could not import a FlashAttention-3 flash_attn_func")

    def forward_partial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.load_function()(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        return _normalize_result(self.name, result, output)


@dataclass(frozen=True)
class Flash4Backend:
    name: str = "fa4"

    @staticmethod
    def load_function():
        from flash_attn.cute import flash_attn_func

        return flash_attn_func

    def forward_partial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.load_function()(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            return_lse=True,
        )
        return _normalize_result(self.name, result, output)


_BACKENDS: dict[str, FlashPartialBackend] = {
    "fa2": Flash2Backend(),
    "fa3": Flash3Backend(),
    "fa4": Flash4Backend(),
}


def get_flash_backend(name: str) -> FlashPartialBackend:
    try:
        return _BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"unsupported FlashAttention backend: {name}") from error


def flash_backend_is_available(name: str) -> bool:
    try:
        backend = get_flash_backend(name)
        if isinstance(backend, Flash2Backend):
            backend.load_extension()
        else:
            backend.load_function()
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def flash_partial_forward(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_flash_backend(backend).forward_partial(
        q,
        k,
        v,
        output,
        softmax_scale=softmax_scale,
    )


__all__ = [
    "FlashPartialBackend",
    "flash_backend_is_available",
    "flash_partial_forward",
    "get_flash_backend",
]
