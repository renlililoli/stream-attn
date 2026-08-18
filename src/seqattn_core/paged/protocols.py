from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import torch

from .layout import KVLayout, PageDescriptor, PageReadMetrics, TensorLayout
from .memory_budget import HostMemoryPlan


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

    def __enter__(self) -> Self:
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
    def open_reader(self, memory_plan: HostMemoryPlan, queue_depth: int) -> PageReader:
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


__all__ = ["PageReader", "PageSink", "PageSource", "PageWriter"]
