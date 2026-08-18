from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Callable, Sequence

import torch

from .host_memory import HostMemoryPlan
from .quantization import quantize_int8_per_token_group


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
    def from_dict(cls, payload: dict[str, int]) -> "PageDescriptor":
        return cls(**payload)


@dataclass(frozen=True)
class PageReadMetrics:
    read_seconds: float = 0.0
    quantization_seconds: float = 0.0
    logical_bytes: int = 0
    physical_bytes: int = 0


def validate_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int, name: str) -> list[int]:
    if cu_seqlens.device.type != "cpu" or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional CPU tensor")
    bounds = cu_seqlens.to(torch.int64).tolist()
    if bounds[0] != 0 or bounds[-1] != total_tokens:
        raise ValueError(f"{name} must span [0, {total_tokens}]")
    if any(stop < start for start, stop in zip(bounds[:-1], bounds[1:])):
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
    for segment_id, (segment_start, segment_stop) in enumerate(
        zip(cu_seqlens[:-1], cu_seqlens[1:])
    ):
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


class PageReader(ABC):
    @abstractmethod
    def read_q(self, page: PageDescriptor, out: torch.Tensor) -> PageReadMetrics:
        raise NotImplementedError

    @abstractmethod
    def read_kv(
        self,
        page: PageDescriptor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k_scales_out: torch.Tensor | None = None,
        v_scales_out: torch.Tensor | None = None,
    ) -> PageReadMetrics:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "PageReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PageSource(ABC):
    q_layout: TensorLayout | None
    kv_layout: KVLayout | None
    q_pages: tuple[PageDescriptor, ...]
    kv_pages: tuple[PageDescriptor, ...]
    cu_seqlens_q: tuple[int, ...] | None
    cu_seqlens_k: tuple[int, ...] | None
    direct_io: bool = False
    backing_kind: str = "memory"

    @abstractmethod
    def open_reader(
        self, memory_plan: HostMemoryPlan, queue_depth: int
    ) -> PageReader:
        raise NotImplementedError


class PageWriter(ABC):
    @abstractmethod
    def write_page(self, page: PageDescriptor, data: torch.Tensor) -> PageReadMetrics:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> object:
        raise NotImplementedError

    def abort(self) -> None:
        return None


class PageSink(ABC):
    backing_kind: str = "memory"

    @abstractmethod
    def open_writer(
        self,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> PageWriter:
        raise NotImplementedError


class _MemoryReader(PageReader):
    def __init__(self, source: "MemoryPageSource") -> None:
        self.source = source

    def read_q(self, page: PageDescriptor, out: torch.Tensor) -> PageReadMetrics:
        if self.source.q is None:
            raise ValueError("this MemoryPageSource has no query tensor")
        started = time.perf_counter()
        source = self.source.q[page.token_start : page.token_stop]
        out[: page.valid_tokens].copy_(source)
        if out.shape[0] > page.valid_tokens:
            out[page.valid_tokens :].zero_()
        elapsed = time.perf_counter() - started
        size = source.numel() * source.element_size()
        return PageReadMetrics(read_seconds=elapsed, logical_bytes=size, physical_bytes=size)

    def read_kv(
        self,
        page: PageDescriptor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k_scales_out: torch.Tensor | None = None,
        v_scales_out: torch.Tensor | None = None,
    ) -> PageReadMetrics:
        if self.source.k is None or self.source.v is None or self.source.kv_layout is None:
            raise ValueError("this MemoryPageSource has no K/V tensors")
        started = time.perf_counter()
        k_source = self.source.k[page.token_start : page.token_stop]
        v_source = self.source.v[page.token_start : page.token_stop]
        quant_seconds = 0.0
        if self.source.kv_layout.storage_dtype == "int8":
            if k_scales_out is None or v_scales_out is None:
                raise ValueError("INT8 K/V reads require scale buffers")
            quant_started = time.perf_counter()
            groups = math.ceil(page.valid_tokens / self.source.kv_layout.quant_group_tokens)
            quantize_int8_per_token_group(
                k_source,
                k_out[: page.valid_tokens],
                k_scales_out[:groups],
                group_tokens=self.source.kv_layout.quant_group_tokens,
            )
            quantize_int8_per_token_group(
                v_source,
                v_out[: page.valid_tokens],
                v_scales_out[:groups],
                group_tokens=self.source.kv_layout.quant_group_tokens,
            )
            quant_seconds = time.perf_counter() - quant_started
        else:
            k_out[: page.valid_tokens].copy_(k_source)
            v_out[: page.valid_tokens].copy_(v_source)
        if k_out.shape[0] > page.valid_tokens:
            k_out[page.valid_tokens :].zero_()
            v_out[page.valid_tokens :].zero_()
        elapsed = time.perf_counter() - started
        logical = 2 * page.valid_tokens * self.source.kv_layout.storage_bytes_per_token
        if self.source.kv_layout.storage_dtype == "int8":
            logical += 2 * math.ceil(
                page.valid_tokens / self.source.kv_layout.quant_group_tokens
            ) * self.source.kv_layout.heads * 2
        return PageReadMetrics(
            read_seconds=elapsed,
            quantization_seconds=quant_seconds,
            logical_bytes=logical,
            physical_bytes=logical,
        )


class MemoryPageSource(PageSource):
    """Page source over caller-owned CPU tensors.

    The complete tensors do not count toward the paged operator's host budget.
    """

    def __init__(
        self,
        *,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        page_target_bytes: int = 16 * 2**20,
        block_n: int = 64,
        kv_storage_dtype: str | None = None,
        quant_group_tokens: int = 64,
    ) -> None:
        if q is None and k is None:
            raise ValueError("at least one of q or k/v must be supplied")
        if (k is None) != (v is None):
            raise ValueError("k and v must be supplied together")
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor is None:
                continue
            if tensor.device.type != "cpu" or tensor.ndim != 3 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous CPU [tokens, heads, head_dim]")
        if k is not None and (k.shape != v.shape or k.dtype != v.dtype):
            raise ValueError("k and v must have matching shape and dtype")
        if q is not None and k is not None:
            if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1]:
                raise ValueError("q and k/v head shapes are incompatible")
            if q.dtype != k.dtype:
                raise ValueError("q and k/v source dtype must match")

        self.q, self.k, self.v = q, k, v
        self.q_layout = None
        self.kv_layout = None
        self.q_pages = ()
        self.kv_pages = ()
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None
        if q is not None:
            if cu_seqlens_q is None:
                raise ValueError("cu_seqlens_q is required with q")
            q_bounds = validate_cu_seqlens(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
            self.cu_seqlens_q = tuple(q_bounds)
            self.q_layout = TensorLayout(q.shape[0], q.shape[1], q.shape[2], dtype_name(q.dtype))
            self.q_pages = build_page_descriptors(
                q_bounds,
                bytes_per_token=self.q_layout.bytes_per_token,
                page_target_bytes=page_target_bytes,
                token_alignment=block_n,
            )
        if k is not None:
            if cu_seqlens_k is None:
                raise ValueError("cu_seqlens_k is required with k/v")
            k_bounds = validate_cu_seqlens(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
            self.cu_seqlens_k = tuple(k_bounds)
            storage = dtype_name(k.dtype) if kv_storage_dtype is None else kv_storage_dtype
            if storage not in {"bf16", "fp16", "fp32", "int8"}:
                raise ValueError("unsupported K/V storage dtype")
            if storage != "int8" and torch_dtype(storage) != k.dtype:
                raise ValueError("exact K/V storage dtype must match the source dtype")
            self.kv_layout = KVLayout(
                k.shape[0],
                k.shape[1],
                k.shape[2],
                dtype_name(k.dtype),
                storage,
                quant_group_tokens,
            )
            self.kv_pages = build_page_descriptors(
                k_bounds,
                bytes_per_token=self.kv_layout.storage_bytes_per_token,
                page_target_bytes=page_target_bytes,
                token_alignment=block_n,
            )

    def open_reader(self, memory_plan: HostMemoryPlan, queue_depth: int) -> PageReader:
        del memory_plan, queue_depth
        return _MemoryReader(self)


class _MemoryPageWriter(PageWriter):
    def __init__(self, sink: "MemoryPageSink", layout: TensorLayout) -> None:
        self.sink = sink
        if sink.out is None:
            raise ValueError("MemoryPageSink requires a caller-owned output tensor")
        if tuple(sink.out.shape) != (layout.total_tokens, layout.heads, layout.head_dim):
            raise ValueError("MemoryPageSink output shape does not match the query layout")
        if sink.out.dtype != layout.torch_dtype or sink.out.device.type != "cpu":
            raise ValueError("MemoryPageSink output must be a matching CPU tensor")

    def write_page(self, page: PageDescriptor, data: torch.Tensor) -> PageReadMetrics:
        started = time.perf_counter()
        self.sink.out[page.token_start : page.token_stop].copy_(data[: page.valid_tokens])
        size = page.valid_tokens * data.shape[1] * data.shape[2] * data.element_size()
        return PageReadMetrics(
            read_seconds=time.perf_counter() - started,
            logical_bytes=size,
            physical_bytes=size,
        )

    def close(self) -> torch.Tensor:
        assert self.sink.out is not None
        return self.sink.out


class MemoryPageSink(PageSink):
    """Compatibility sink writing into a caller-owned complete CPU tensor."""

    def __init__(self, out: torch.Tensor) -> None:
        self.out = out

    def open_writer(
        self,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> PageWriter:
        del cu_seqlens, pages, memory_plan, queue_depth
        return _MemoryPageWriter(self, layout)


class _CallbackPageWriter(PageWriter):
    def __init__(
        self,
        callback: Callable[[PageDescriptor, torch.Tensor], object],
    ) -> None:
        self.callback = callback
        self.pages_written = 0

    def write_page(self, page: PageDescriptor, data: torch.Tensor) -> PageReadMetrics:
        started = time.perf_counter()
        self.callback(page, data[: page.valid_tokens])
        self.pages_written += 1
        size = page.valid_tokens * data.shape[1] * data.shape[2] * data.element_size()
        return PageReadMetrics(
            read_seconds=time.perf_counter() - started,
            logical_bytes=size,
            physical_bytes=size,
        )

    def close(self) -> int:
        return self.pages_written


class CallbackOutputSink(PageSink):
    """Synchronously hands each output page to a caller callback."""

    def __init__(self, callback: Callable[[PageDescriptor, torch.Tensor], object]) -> None:
        self.callback = callback

    def open_writer(
        self,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> PageWriter:
        del layout, cu_seqlens, pages, memory_plan, queue_depth
        return _CallbackPageWriter(self.callback)


def replace_page(page: PageDescriptor, **changes: int) -> PageDescriptor:
    return replace(page, **changes)


__all__ = [
    "CallbackOutputSink",
    "KVLayout",
    "MemoryPageSink",
    "MemoryPageSource",
    "PageDescriptor",
    "PageReadMetrics",
    "PageReader",
    "PageSink",
    "PageSource",
    "PageWriter",
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
