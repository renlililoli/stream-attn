from __future__ import annotations

import math
import mmap
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch

from .layout import KVLayout, PageDescriptor, align_up
from .memory_budget import HostMemoryPlan


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    seconds: float


class KVPageCache:
    """Preallocated two-region K/V cache with a deterministic low-id hot set."""

    def __init__(
        self,
        pages: Sequence[PageDescriptor],
        layout: KVLayout,
        *,
        capacity_bytes: int,
        hot_fraction: float,
        memory_plan: HostMemoryPlan,
    ) -> None:
        self.pages = tuple(pages)
        self.layout = layout
        self.max_tokens = max((page.padded_tokens for page in pages), default=0)
        self.max_groups = (
            math.ceil(self.max_tokens / layout.quant_group_tokens)
            if layout.storage_dtype == "int8"
            else 0
        )
        self.tensor_bytes = self.max_tokens * layout.storage_bytes_per_token
        self.scale_bytes = self.max_groups * layout.heads * 2
        self.k_offset = 0
        self.v_offset = self.tensor_bytes
        self.k_scale_offset = 2 * self.tensor_bytes
        self.v_scale_offset = self.k_scale_offset + self.scale_bytes
        self.slot_bytes = (
            align_up(
                2 * self.tensor_bytes + 2 * self.scale_bytes,
                64,
            )
            if self.max_tokens
            else 0
        )
        self.slot_count = (
            min(len(pages), capacity_bytes // self.slot_bytes) if self.slot_bytes else 0
        )
        self.registered_bytes = self.slot_count * self.slot_bytes
        self.memory_plan = memory_plan
        self.mapping: mmap.mmap | None = None
        self.raw = torch.empty(0, dtype=torch.uint8)
        if self.registered_bytes:
            memory_plan.register("cache", self.registered_bytes)
            try:
                self.mapping = mmap.mmap(-1, self.registered_bytes, access=mmap.ACCESS_WRITE)
                self.raw = torch.frombuffer(
                    self.mapping, dtype=torch.uint8, count=self.registered_bytes
                )
            except BaseException:
                memory_plan.release("cache", self.registered_bytes)
                raise

        hot_slots = int(self.slot_count * hot_fraction)
        if hot_fraction > 0 and self.slot_count:
            hot_slots = max(1, hot_slots)
        self.hot_slot_count = min(self.slot_count, hot_slots)
        self.rolling_slot_count = self.slot_count - self.hot_slot_count
        self.hot_page_ids = frozenset(page.page_id for page in pages[: self.hot_slot_count])
        self._hot_present: set[int] = set()
        self._rolling: OrderedDict[int, int] = OrderedDict()
        self._rolling_free = deque(
            range(self.hot_slot_count, self.hot_slot_count + self.rolling_slot_count)
        )
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0
        self._present_count = 0
        self._lock = threading.Lock()

    def _slot_slice(self, slot: int, offset: int, size: int) -> torch.Tensor:
        start = slot * self.slot_bytes + offset
        return self.raw[start : start + size]

    @staticmethod
    def _copy_tensor_to_raw(target: torch.Tensor, source: torch.Tensor) -> None:
        bytes_source = source.detach().contiguous().view(torch.uint8).reshape(-1)
        target[: bytes_source.numel()].copy_(bytes_source)

    @staticmethod
    def _copy_raw_to_tensor(source: torch.Tensor, target: torch.Tensor) -> None:
        target.view(torch.uint8).reshape(-1).copy_(source[: target.numel() * target.element_size()])

    def _find_slot(self, page_id: int) -> int | None:
        if page_id in self.hot_page_ids:
            return page_id if page_id in self._hot_present else None
        slot = self._rolling.get(page_id)
        if slot is not None:
            self._rolling.move_to_end(page_id)
        return slot

    def get(
        self,
        page: PageDescriptor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k_scales_out: torch.Tensor | None = None,
        v_scales_out: torch.Tensor | None = None,
    ) -> CacheLookup:
        started = time.perf_counter()
        with self._lock:
            slot = self._find_slot(page.page_id)
            if slot is None:
                self.misses += 1
                return CacheLookup(False, time.perf_counter() - started)
            tensor_bytes = page.valid_tokens * self.layout.storage_bytes_per_token
            self._copy_raw_to_tensor(
                self._slot_slice(slot, self.k_offset, tensor_bytes),
                k_out[: page.valid_tokens],
            )
            self._copy_raw_to_tensor(
                self._slot_slice(slot, self.v_offset, tensor_bytes),
                v_out[: page.valid_tokens],
            )
            if self.layout.storage_dtype == "int8":
                if k_scales_out is None or v_scales_out is None:
                    raise ValueError("INT8 cache reads require scale buffers")
                groups = math.ceil(page.valid_tokens / self.layout.quant_group_tokens)
                scale_bytes = groups * self.layout.heads * 2
                self._copy_raw_to_tensor(
                    self._slot_slice(slot, self.k_scale_offset, scale_bytes),
                    k_scales_out[:groups],
                )
                self._copy_raw_to_tensor(
                    self._slot_slice(slot, self.v_scale_offset, scale_bytes),
                    v_scales_out[:groups],
                )
            if k_out.shape[0] > page.valid_tokens:
                k_out[page.valid_tokens :].zero_()
                v_out[page.valid_tokens :].zero_()
            self.hits += 1
        return CacheLookup(True, time.perf_counter() - started)

    def put(
        self,
        page: PageDescriptor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
    ) -> bool:
        if not self.slot_count:
            return False
        with self._lock:
            existing = self._find_slot(page.page_id)
            if existing is not None:
                return True
            if page.page_id in self.hot_page_ids:
                slot = page.page_id
                self._hot_present.add(page.page_id)
            elif self.rolling_slot_count:
                if self._rolling_free:
                    slot = self._rolling_free.popleft()
                    self._present_count += 1
                else:
                    _, slot = self._rolling.popitem(last=False)
                self._rolling[page.page_id] = slot
            else:
                return False

            if page.page_id in self.hot_page_ids:
                self._present_count += 1
            tensor_bytes = page.valid_tokens * self.layout.storage_bytes_per_token
            self._copy_tensor_to_raw(
                self._slot_slice(slot, self.k_offset, tensor_bytes),
                k[: page.valid_tokens],
            )
            self._copy_tensor_to_raw(
                self._slot_slice(slot, self.v_offset, tensor_bytes),
                v[: page.valid_tokens],
            )
            if self.layout.storage_dtype == "int8":
                if k_scales is None or v_scales is None:
                    raise ValueError("INT8 cache inserts require scale buffers")
                groups = math.ceil(page.valid_tokens / self.layout.quant_group_tokens)
                scale_bytes = groups * self.layout.heads * 2
                self._copy_tensor_to_raw(
                    self._slot_slice(slot, self.k_scale_offset, scale_bytes),
                    k_scales[:groups],
                )
                self._copy_tensor_to_raw(
                    self._slot_slice(slot, self.v_scale_offset, scale_bytes),
                    v_scales[:groups],
                )
            self.peak_bytes = max(
                self.peak_bytes,
                min(self._present_count, self.slot_count) * self.slot_bytes,
            )
            return True

    def close(self) -> None:
        if self.mapping is None:
            return
        self.raw = torch.empty(0, dtype=torch.uint8)
        self.mapping.close()
        self.mapping = None
        self.memory_plan.release("cache", self.registered_bytes)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["CacheLookup", "KVPageCache"]
