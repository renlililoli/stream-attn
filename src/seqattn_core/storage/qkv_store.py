from __future__ import annotations

import errno
import json
import math
import os
import queue
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from ..paged.layout import (
    KVLayout,
    PageDescriptor,
    PageReadMetrics,
    TensorLayout,
    pages_by_segment,
)
from ..paged.memory_budget import HostMemoryPlan
from ..paged.protocols import PageReader, PageSource
from .direct_io import DIRECT_IO_ALIGNMENT, FORMAT_VERSION, AlignedBuffer, _open_file
from .records import expected_file_size


class _NvmePageReader(PageReader):
    def __init__(
        self,
        store: NvmeQKVStore,
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> None:
        self.store = store
        self.direct_io = store.direct_io
        max_record = max(
            [page.storage_bytes for page in (*store.q_pages, *store.kv_pages)]
            or [DIRECT_IO_ALIGNMENT]
        )
        self.buffers = [AlignedBuffer(max_record) for _ in range(queue_depth)]
        self.registered_bytes = sum(buffer.size_bytes for buffer in self.buffers)
        try:
            memory_plan.register("bounce", self.registered_bytes)
        except BaseException:
            for buffer in self.buffers:
                buffer.close()
            raise
        self.memory_plan = memory_plan
        self.available: queue.LifoQueue[AlignedBuffer] = queue.LifoQueue()
        for buffer in self.buffers:
            self.available.put(buffer)
        try:
            self.q_fd = _open_file(store.q_path, os.O_RDONLY, 0, self.direct_io)
            self.kv_fd = _open_file(store.kv_path, os.O_RDONLY, 0, self.direct_io)
        except BaseException:
            if hasattr(self, "q_fd"):
                os.close(self.q_fd)
            self.memory_plan.release("bounce", self.registered_bytes)
            for buffer in self.buffers:
                buffer.close()
            raise
        self.closed = False

    def _read(self, fd: int, path: Path, page: PageDescriptor) -> tuple[AlignedBuffer, float]:
        buffer = self.available.get()
        started = time.perf_counter()
        view = memoryview(buffer.mapping)[: page.storage_bytes]
        try:
            read = os.preadv(fd, [view], page.file_offset)
        except OSError as error:
            self.available.put(buffer)
            if self.direct_io and error.errno in {
                errno.EINVAL,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,
            }:
                raise RuntimeError(
                    f"filesystem rejected required O_DIRECT read for {path}"
                ) from error
            raise
        finally:
            view.release()
        if read != page.storage_bytes:
            self.available.put(buffer)
            raise OSError(errno.EIO, f"short read: {read} != {page.storage_bytes} from {path}")
        return buffer, time.perf_counter() - started

    def read_q(self, page: PageDescriptor, out: torch.Tensor) -> PageReadMetrics:
        buffer, elapsed = self._read(self.q_fd, self.store.q_path, page)
        try:
            elements = page.valid_tokens * self.store.q_layout.heads * self.store.q_layout.head_dim
            buffer.copy_to_tensor(0, out, elements)
            if out.shape[0] > page.valid_tokens:
                out[page.valid_tokens :].zero_()
        finally:
            self.available.put(buffer)
        logical = page.valid_tokens * self.store.q_layout.bytes_per_token
        return PageReadMetrics(
            read_seconds=elapsed,
            logical_bytes=logical,
            physical_bytes=page.storage_bytes,
        )

    def read_kv(
        self,
        page: PageDescriptor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k_scales_out: torch.Tensor | None = None,
        v_scales_out: torch.Tensor | None = None,
    ) -> PageReadMetrics:
        buffer, elapsed = self._read(self.kv_fd, self.store.kv_path, page)
        try:
            elements = (
                page.valid_tokens * self.store.kv_layout.heads * self.store.kv_layout.head_dim
            )
            buffer.copy_to_tensor(0, k_out, elements)
            buffer.copy_to_tensor(page.v_offset, v_out, elements)
            logical = 2 * elements * k_out.element_size()
            if self.store.kv_layout.storage_dtype == "int8":
                if k_scales_out is None or v_scales_out is None:
                    raise ValueError("INT8 K/V reads require scale buffers")
                groups = math.ceil(page.valid_tokens / self.store.kv_layout.quant_group_tokens)
                scale_elements = groups * self.store.kv_layout.heads
                buffer.copy_to_tensor(page.k_scale_offset, k_scales_out, scale_elements)
                buffer.copy_to_tensor(page.v_scale_offset, v_scales_out, scale_elements)
                logical += 2 * scale_elements * 2
            if k_out.shape[0] > page.valid_tokens:
                k_out[page.valid_tokens :].zero_()
                v_out[page.valid_tokens :].zero_()
        finally:
            self.available.put(buffer)
        return PageReadMetrics(
            read_seconds=elapsed,
            logical_bytes=logical,
            physical_bytes=page.storage_bytes,
        )

    def close(self) -> None:
        if self.closed:
            return
        os.close(self.q_fd)
        os.close(self.kv_fd)
        self.memory_plan.release("bounce", self.registered_bytes)
        for buffer in self.buffers:
            buffer.close()
        self.closed = True


class NvmeQKVStore(PageSource):
    backing_kind = "nvme"

    def __init__(self, path: str | os.PathLike[str], *, direct_io: bool = True) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read valid store manifest: {manifest_path}") from error
        if manifest.get("format") != "seqattn-qkv" or manifest.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported or corrupt seqattn Q/K/V manifest")
        if manifest.get("alignment_bytes") != DIRECT_IO_ALIGNMENT:
            raise ValueError("manifest alignment is incompatible with this runtime")
        try:
            self.q_layout = TensorLayout(**manifest["q_layout"])
            self.kv_layout = KVLayout(**manifest["kv_layout"])
            self.cu_seqlens_q = tuple(int(value) for value in manifest["cu_seqlens_q"])
            self.cu_seqlens_k = tuple(int(value) for value in manifest["cu_seqlens_k"])
            self.q_pages = tuple(PageDescriptor.from_dict(page) for page in manifest["q_pages"])
            self.kv_pages = tuple(PageDescriptor.from_dict(page) for page in manifest["kv_pages"])
            self.q_path = self.path / manifest["files"]["q"]["name"]
            self.kv_path = self.path / manifest["files"]["kv"]["name"]
            q_size = int(manifest["files"]["q"]["size_bytes"])
            kv_size = int(manifest["files"]["kv"]["size_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("manifest is missing required Q/K/V layout fields") from error
        self.direct_io = direct_io
        self.quantization_seconds = float(manifest.get("quantization_seconds", 0.0))
        self._validate_pages(self.q_pages, self.cu_seqlens_q, q_size, "q")
        self._validate_pages(self.kv_pages, self.cu_seqlens_k, kv_size, "kv")
        for data_path, expected in ((self.q_path, q_size), (self.kv_path, kv_size)):
            try:
                actual = data_path.stat().st_size
            except OSError as error:
                raise ValueError(f"manifest data file is missing: {data_path}") from error
            if actual != expected:
                raise ValueError(
                    f"manifest size mismatch for {data_path.name}: {actual} != {expected}"
                )

    @staticmethod
    def _validate_pages(
        pages: Sequence[PageDescriptor],
        cu_seqlens: Sequence[int],
        file_size: int,
        name: str,
    ) -> None:
        grouped = pages_by_segment(pages, len(cu_seqlens) - 1)
        expected_id = 0
        for segment_id, group in enumerate(grouped):
            expected_token = cu_seqlens[segment_id]
            for page in group:
                if page.page_id != expected_id:
                    raise ValueError(f"{name} page ids are not contiguous")
                if page.token_start != expected_token or page.segment_id != segment_id:
                    raise ValueError(f"{name} page table crosses or skips a segment boundary")
                if page.valid_tokens <= 0 or page.padded_tokens < page.valid_tokens:
                    raise ValueError(f"{name} page has invalid token counts")
                if (
                    page.file_offset % DIRECT_IO_ALIGNMENT
                    or page.storage_bytes % DIRECT_IO_ALIGNMENT
                ):
                    raise ValueError(f"{name} page is not direct-I/O aligned")
                if page.payload_bytes + page.padding_bytes != page.storage_bytes:
                    raise ValueError(f"{name} page padding metadata is inconsistent")
                expected_token = page.token_stop
                expected_id += 1
            if expected_token != cu_seqlens[segment_id + 1]:
                raise ValueError(f"{name} pages do not cover segment {segment_id}")
        if expected_file_size(pages) != file_size:
            raise ValueError(f"{name} page table does not match file size")

    def open_reader(self, memory_plan: HostMemoryPlan, queue_depth: int) -> PageReader:
        return _NvmePageReader(self, memory_plan, queue_depth)


__all__ = ["NvmeQKVStore"]
