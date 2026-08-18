from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

import torch

DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.int8: "int8",
}
NAME_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return value - value % alignment


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return DTYPE_NAMES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {dtype}") from error


def torch_dtype(name: str) -> torch.dtype:
    try:
        return NAME_DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype name: {name}") from error


@dataclass(frozen=True)
class TensorLayout:
    total_tokens: int
    heads: int
    head_dim: int
    dtype: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.dtype)

    @property
    def bytes_per_token(self) -> int:
        return self.heads * self.head_dim * torch.empty((), dtype=self.torch_dtype).element_size()

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class KVLayout:
    total_tokens: int
    heads: int
    head_dim: int
    source_dtype: str
    storage_dtype: str
    quant_group_tokens: int = 64

    @property
    def source_torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.source_dtype)

    @property
    def storage_torch_dtype(self) -> torch.dtype:
        return torch_dtype(self.storage_dtype)

    @property
    def storage_bytes_per_token(self) -> int:
        return (
            self.heads
            * self.head_dim
            * torch.empty((), dtype=self.storage_torch_dtype).element_size()
        )

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PageDescriptor:
    page_id: int
    segment_id: int
    token_start: int
    segment_token_start: int
    valid_tokens: int
    padded_tokens: int
    file_offset: int = 0
    payload_bytes: int = 0
    storage_bytes: int = 0
    padding_bytes: int = 0
    k_bytes: int = 0
    v_offset: int = 0
    v_bytes: int = 0
    k_scale_offset: int = 0
    v_scale_offset: int = 0
    scale_bytes: int = 0

    @property
    def token_stop(self) -> int:
        return self.token_start + self.valid_tokens

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> PageDescriptor:
        return cls(**payload)


@dataclass(frozen=True)
class PageReadMetrics:
    read_seconds: float = 0.0
    quantization_seconds: float = 0.0
    logical_bytes: int = 0
    physical_bytes: int = 0
    simulated_io_seconds: float = 0.0
    simulated_service_seconds: float = 0.0
    simulated_queue_seconds: float = 0.0
    simulated_logical_bytes: int = 0
    simulated_physical_bytes: int = 0


def validate_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int, name: str) -> list[int]:
    if cu_seqlens.device.type != "cpu" or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional CPU tensor")
    bounds = cu_seqlens.to(torch.int64).tolist()
    if bounds[0] != 0 or bounds[-1] != total_tokens:
        raise ValueError(f"{name} must span [0, {total_tokens}]")
    if any(stop < start for start, stop in pairwise(bounds)):
        raise ValueError(f"{name} must be non-decreasing")
    return bounds


def build_page_descriptors(
    cu_seqlens: Sequence[int],
    *,
    bytes_per_token: int,
    page_target_bytes: int,
    token_alignment: int,
) -> tuple[PageDescriptor, ...]:
    if bytes_per_token <= 0 or page_target_bytes <= 0:
        raise ValueError("page sizing inputs must be positive")
    target_tokens = max(token_alignment, page_target_bytes // bytes_per_token)
    target_tokens = max(token_alignment, align_down(target_tokens, token_alignment))
    pages: list[PageDescriptor] = []
    for segment_id, (segment_start, segment_stop) in enumerate(pairwise(cu_seqlens)):
        local_start = 0
        while segment_start + local_start < segment_stop:
            valid = min(target_tokens, segment_stop - segment_start - local_start)
            padded = align_up(valid, token_alignment)
            pages.append(
                PageDescriptor(
                    page_id=len(pages),
                    segment_id=segment_id,
                    token_start=segment_start + local_start,
                    segment_token_start=local_start,
                    valid_tokens=valid,
                    padded_tokens=padded,
                )
            )
            local_start += valid
    return tuple(pages)


def pages_by_segment(
    pages: Sequence[PageDescriptor], segment_count: int
) -> tuple[tuple[PageDescriptor, ...], ...]:
    grouped: list[list[PageDescriptor]] = [[] for _ in range(segment_count)]
    for page in pages:
        if page.segment_id < 0 or page.segment_id >= segment_count:
            raise ValueError("page references an invalid segment")
        grouped[page.segment_id].append(page)
    return tuple(tuple(group) for group in grouped)


def replace_page(page: PageDescriptor, **changes: int) -> PageDescriptor:
    return replace(page, **changes)


__all__ = [
    "KVLayout",
    "PageDescriptor",
    "PageReadMetrics",
    "TensorLayout",
    "align_down",
    "align_up",
    "build_page_descriptors",
    "dtype_name",
    "pages_by_segment",
    "replace_page",
    "torch_dtype",
    "validate_cu_seqlens",
]
