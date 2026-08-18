from __future__ import annotations

import errno
import json
import math
import mmap
import os
import queue
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch

from .host_memory import HostMemoryPlan
from .paging import (
    KVLayout,
    PageDescriptor,
    PageReadMetrics,
    PageReader,
    PageSink,
    PageSource,
    PageWriter,
    TensorLayout,
    align_up,
    build_page_descriptors,
    pages_by_segment,
    replace_page,
)
from .quantization import quantize_int8_per_token_group


DIRECT_IO_ALIGNMENT = 4096
FORMAT_VERSION = 1


def _direct_flag() -> int:
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        raise RuntimeError("O_DIRECT is unavailable on this platform")
    return flag


def _open_file(path: Path, flags: int, mode: int, direct_io: bool) -> int:
    if direct_io:
        flags |= _direct_flag()
    try:
        return os.open(path, flags, mode)
    except OSError as error:
        if direct_io and error.errno in {
            errno.EINVAL,
            errno.EOPNOTSUPP,
            errno.ENOTSUP,
            errno.ENODEV,
        }:
            raise RuntimeError(
                f"filesystem does not support required O_DIRECT access for {path}"
            ) from error
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(errno.EIO, f"short manifest write: {written} != {len(encoded)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class _AlignedBuffer:
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = align_up(max(size_bytes, DIRECT_IO_ALIGNMENT), DIRECT_IO_ALIGNMENT)
        self.mapping = mmap.mmap(-1, self.size_bytes, access=mmap.ACCESS_WRITE)
        self.bytes = torch.frombuffer(self.mapping, dtype=torch.uint8, count=self.size_bytes)

    def zero(self, size_bytes: int) -> None:
        self.bytes[:size_bytes].zero_()

    def copy_tensor(self, offset: int, tensor: torch.Tensor) -> int:
        source = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
        stop = offset + source.numel()
        if stop > self.size_bytes:
            raise ValueError("tensor does not fit in the aligned bounce buffer")
        self.bytes[offset:stop].copy_(source)
        return source.numel()

    def copy_to_tensor(
        self,
        offset: int,
        tensor: torch.Tensor,
        element_count: int,
    ) -> None:
        byte_count = element_count * tensor.element_size()
        source = self.bytes[offset : offset + byte_count]
        tensor.reshape(-1)[:element_count].view(torch.uint8).copy_(source)

    def close(self) -> None:
        self.bytes = torch.empty(0, dtype=torch.uint8)
        self.mapping.close()


class _AlignedFileWriter:
    def __init__(self, path: Path, max_record_bytes: int, direct_io: bool) -> None:
        self.path = path
        self.direct_io = direct_io
        self.fd = _open_file(
            path,
            os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
            0o644,
            direct_io,
        )
        self.buffer = _AlignedBuffer(max_record_bytes)
        self.closed = False

    def write(self, offset: int, storage_bytes: int) -> None:
        if offset % DIRECT_IO_ALIGNMENT or storage_bytes % DIRECT_IO_ALIGNMENT:
            raise ValueError("direct-I/O writes must use 4096-byte aligned offsets and lengths")
        view = memoryview(self.buffer.mapping)[:storage_bytes]
        try:
            written = os.pwritev(self.fd, [view], offset)
        except OSError as error:
            if self.direct_io and error.errno in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise RuntimeError(
                    f"filesystem rejected required O_DIRECT write for {self.path}"
                ) from error
            raise
        finally:
            view.release()
        if written != storage_bytes:
            raise OSError(errno.EIO, f"short write: {written} != {storage_bytes}")

    def close(self) -> None:
        if self.closed:
            return
        try:
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.buffer.close()
            self.closed = True


def _q_record_pages(
    pages: Sequence[PageDescriptor], layout: TensorLayout
) -> tuple[PageDescriptor, ...]:
    offset = 0
    result = []
    for page in pages:
        payload = page.padded_tokens * layout.bytes_per_token
        storage = align_up(payload, DIRECT_IO_ALIGNMENT)
        result.append(
            replace_page(
                page,
                file_offset=offset,
                payload_bytes=payload,
                storage_bytes=storage,
                padding_bytes=storage - payload,
            )
        )
        offset += storage
    return tuple(result)


def _kv_record_pages(
    pages: Sequence[PageDescriptor], layout: KVLayout
) -> tuple[PageDescriptor, ...]:
    offset = 0
    result = []
    for page in pages:
        tensor_bytes = page.padded_tokens * layout.storage_bytes_per_token
        v_offset = tensor_bytes
        payload = tensor_bytes * 2
        k_scale_offset = 0
        v_scale_offset = 0
        scale_bytes = 0
        if layout.storage_dtype == "int8":
            groups = math.ceil(page.padded_tokens / layout.quant_group_tokens)
            scale_bytes = groups * layout.heads * 2
            k_scale_offset = payload
            v_scale_offset = k_scale_offset + scale_bytes
            payload += 2 * scale_bytes
        storage = align_up(payload, DIRECT_IO_ALIGNMENT)
        result.append(
            replace_page(
                page,
                file_offset=offset,
                payload_bytes=payload,
                storage_bytes=storage,
                padding_bytes=storage - payload,
                k_bytes=tensor_bytes,
                v_offset=v_offset,
                v_bytes=tensor_bytes,
                k_scale_offset=k_scale_offset,
                v_scale_offset=v_scale_offset,
                scale_bytes=scale_bytes,
            )
        )
        offset += storage
    return tuple(result)


def _expected_file_size(pages: Sequence[PageDescriptor]) -> int:
    if not pages:
        return 0
    last = pages[-1]
    return last.file_offset + last.storage_bytes


class NvmeQKVWriter:
    """Streaming builder for an aligned, manifest-described Q/K/V store."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        q_layout: TensorLayout,
        kv_layout: KVLayout,
        cu_seqlens_q: Sequence[int],
        cu_seqlens_k: Sequence[int],
        page_target_bytes: int = 16 * 2**20,
        block_n: int = 64,
        direct_io: bool = True,
    ) -> None:
        self.path = Path(path)
        self.q_layout = q_layout
        self.kv_layout = kv_layout
        self.cu_seqlens_q = tuple(int(value) for value in cu_seqlens_q)
        self.cu_seqlens_k = tuple(int(value) for value in cu_seqlens_k)
        self.page_target_bytes = page_target_bytes
        self.block_n = block_n
        self.direct_io = direct_io
        self.quantization_seconds = 0.0
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "manifest.json").exists():
            raise FileExistsError(f"store already exists: {self.path}")
        self.q_tmp = self.path / "q.bin.tmp"
        self.kv_tmp = self.path / "kv.bin.tmp"
        self._temporary_paths = [self.q_tmp, self.kv_tmp, self.path / "manifest.json.tmp"]

        q_base = build_page_descriptors(
            self.cu_seqlens_q,
            bytes_per_token=q_layout.bytes_per_token,
            page_target_bytes=page_target_bytes,
            token_alignment=block_n,
        )
        kv_base = build_page_descriptors(
            self.cu_seqlens_k,
            bytes_per_token=kv_layout.storage_bytes_per_token,
            page_target_bytes=page_target_bytes,
            token_alignment=block_n,
        )
        self.q_pages = _q_record_pages(q_base, q_layout)
        self.kv_pages = _kv_record_pages(kv_base, kv_layout)

    def _validate_q_page(self, page: PageDescriptor, tensor: torch.Tensor) -> None:
        expected = (page.valid_tokens, self.q_layout.heads, self.q_layout.head_dim)
        if tensor.device.type != "cpu" or tuple(tensor.shape) != expected:
            raise ValueError(f"query page {page.page_id} must use CPU shape {expected}")
        if tensor.dtype != self.q_layout.torch_dtype:
            raise ValueError("query page dtype does not match q_layout")

    def _validate_kv_page(
        self, page: PageDescriptor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        expected = (page.valid_tokens, self.kv_layout.heads, self.kv_layout.head_dim)
        if any(tensor.device.type != "cpu" for tensor in (k, v)):
            raise ValueError("K/V writer pages must be CPU tensors")
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"K/V page {page.page_id} must use shape {expected}")
        if k.dtype != self.kv_layout.source_torch_dtype or v.dtype != k.dtype:
            raise ValueError("K/V page dtype does not match kv_layout")

    def write_pages(
        self,
        q_pages: Iterable[torch.Tensor],
        kv_pages: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> "NvmeQKVStore":
        q_iterator = iter(q_pages)
        kv_iterator = iter(kv_pages)
        q_max = max((page.storage_bytes for page in self.q_pages), default=DIRECT_IO_ALIGNMENT)
        kv_max = max((page.storage_bytes for page in self.kv_pages), default=DIRECT_IO_ALIGNMENT)
        q_writer = None
        kv_writer = None
        try:
            q_writer = _AlignedFileWriter(self.q_tmp, q_max, self.direct_io)
            kv_writer = _AlignedFileWriter(self.kv_tmp, kv_max, self.direct_io)
            for page in self.q_pages:
                try:
                    tensor = next(q_iterator)
                except StopIteration as error:
                    raise ValueError("query page iterator ended early") from error
                self._validate_q_page(page, tensor)
                q_writer.buffer.zero(page.storage_bytes)
                q_writer.buffer.copy_tensor(0, tensor)
                q_writer.write(page.file_offset, page.storage_bytes)
            try:
                next(q_iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("query page iterator yielded extra pages")

            max_tokens = max((page.padded_tokens for page in self.kv_pages), default=0)
            if self.kv_layout.storage_dtype == "int8" and max_tokens:
                k_quantized = torch.empty(
                    (max_tokens, self.kv_layout.heads, self.kv_layout.head_dim),
                    dtype=torch.int8,
                )
                v_quantized = torch.empty_like(k_quantized)
                max_groups = math.ceil(max_tokens / self.kv_layout.quant_group_tokens)
                k_scales = torch.empty((max_groups, self.kv_layout.heads), dtype=torch.float16)
                v_scales = torch.empty_like(k_scales)
            else:
                k_quantized = v_quantized = k_scales = v_scales = None

            for page in self.kv_pages:
                try:
                    k, v = next(kv_iterator)
                except StopIteration as error:
                    raise ValueError("K/V page iterator ended early") from error
                self._validate_kv_page(page, k, v)
                kv_writer.buffer.zero(page.storage_bytes)
                if self.kv_layout.storage_dtype == "int8":
                    assert k_quantized is not None and v_quantized is not None
                    assert k_scales is not None and v_scales is not None
                    groups = math.ceil(page.valid_tokens / self.kv_layout.quant_group_tokens)
                    started = time.perf_counter()
                    quantize_int8_per_token_group(
                        k,
                        k_quantized[: page.valid_tokens],
                        k_scales[:groups],
                        group_tokens=self.kv_layout.quant_group_tokens,
                    )
                    quantize_int8_per_token_group(
                        v,
                        v_quantized[: page.valid_tokens],
                        v_scales[:groups],
                        group_tokens=self.kv_layout.quant_group_tokens,
                    )
                    self.quantization_seconds += time.perf_counter() - started
                    kv_writer.buffer.copy_tensor(0, k_quantized[: page.valid_tokens])
                    kv_writer.buffer.copy_tensor(page.v_offset, v_quantized[: page.valid_tokens])
                    kv_writer.buffer.copy_tensor(page.k_scale_offset, k_scales[:groups])
                    kv_writer.buffer.copy_tensor(page.v_scale_offset, v_scales[:groups])
                else:
                    kv_writer.buffer.copy_tensor(0, k)
                    kv_writer.buffer.copy_tensor(page.v_offset, v)
                kv_writer.write(page.file_offset, page.storage_bytes)
            try:
                next(kv_iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("K/V page iterator yielded extra pages")

            q_writer.close()
            kv_writer.close()
            q_writer = kv_writer = None
            if self.q_tmp.stat().st_size != _expected_file_size(self.q_pages):
                raise OSError(errno.EIO, "query file size does not match its page table")
            if self.kv_tmp.stat().st_size != _expected_file_size(self.kv_pages):
                raise OSError(errno.EIO, "K/V file size does not match its page table")
            os.replace(self.q_tmp, self.path / "q.bin")
            os.replace(self.kv_tmp, self.path / "kv.bin")
            _fsync_directory(self.path)
            manifest = {
                "format": "seqattn-qkv",
                "version": FORMAT_VERSION,
                "alignment_bytes": DIRECT_IO_ALIGNMENT,
                "page_target_bytes": self.page_target_bytes,
                "block_n": self.block_n,
                "q_layout": self.q_layout.as_dict(),
                "kv_layout": self.kv_layout.as_dict(),
                "cu_seqlens_q": list(self.cu_seqlens_q),
                "cu_seqlens_k": list(self.cu_seqlens_k),
                "q_pages": [page.as_dict() for page in self.q_pages],
                "kv_pages": [page.as_dict() for page in self.kv_pages],
                "files": {
                    "q": {"name": "q.bin", "size_bytes": _expected_file_size(self.q_pages)},
                    "kv": {"name": "kv.bin", "size_bytes": _expected_file_size(self.kv_pages)},
                },
                "quantization_seconds": self.quantization_seconds,
            }
            _atomic_write_json(self.path / "manifest.json", manifest)
            return NvmeQKVStore(self.path, direct_io=self.direct_io)
        except BaseException:
            if q_writer is not None:
                q_writer.close()
            if kv_writer is not None:
                kv_writer.close()
            self.abort()
            raise

    def abort(self) -> None:
        paths = list(self._temporary_paths)
        if not (self.path / "manifest.json").exists():
            paths.extend((self.path / "q.bin", self.path / "kv.bin"))
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @classmethod
    def from_tensors(
        cls,
        path: str | os.PathLike[str],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        *,
        page_target_bytes: int = 16 * 2**20,
        block_n: int = 64,
        kv_storage_dtype: str | None = None,
        quant_group_tokens: int = 64,
        direct_io: bool = True,
    ) -> "NvmeQKVStore":
        from .paging import MemoryPageSource

        source = MemoryPageSource(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_target_bytes=page_target_bytes,
            block_n=block_n,
            kv_storage_dtype=kv_storage_dtype,
            quant_group_tokens=quant_group_tokens,
        )
        assert source.q_layout is not None and source.kv_layout is not None
        writer = cls(
            path,
            q_layout=source.q_layout,
            kv_layout=source.kv_layout,
            cu_seqlens_q=source.cu_seqlens_q or (),
            cu_seqlens_k=source.cu_seqlens_k or (),
            page_target_bytes=page_target_bytes,
            block_n=block_n,
            direct_io=direct_io,
        )
        q_iter = (q[page.token_start : page.token_stop] for page in writer.q_pages)
        kv_iter = (
            (k[page.token_start : page.token_stop], v[page.token_start : page.token_stop])
            for page in writer.kv_pages
        )
        return writer.write_pages(q_iter, kv_iter)


class _NvmePageReader(PageReader):
    def __init__(
        self,
        store: "NvmeQKVStore",
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> None:
        self.store = store
        self.direct_io = store.direct_io
        max_record = max(
            [page.storage_bytes for page in (*store.q_pages, *store.kv_pages)]
            or [DIRECT_IO_ALIGNMENT]
        )
        self.buffers = [_AlignedBuffer(max_record) for _ in range(queue_depth)]
        self.registered_bytes = sum(buffer.size_bytes for buffer in self.buffers)
        try:
            memory_plan.register("bounce", self.registered_bytes)
        except BaseException:
            for buffer in self.buffers:
                buffer.close()
            raise
        self.memory_plan = memory_plan
        self.available: queue.LifoQueue[_AlignedBuffer] = queue.LifoQueue()
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

    def _read(self, fd: int, path: Path, page: PageDescriptor) -> tuple[_AlignedBuffer, float]:
        buffer = self.available.get()
        started = time.perf_counter()
        view = memoryview(buffer.mapping)[: page.storage_bytes]
        try:
            read = os.preadv(fd, [view], page.file_offset)
        except OSError as error:
            self.available.put(buffer)
            if self.direct_io and error.errno in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
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
                page.valid_tokens
                * self.store.kv_layout.heads
                * self.store.kv_layout.head_dim
            )
            buffer.copy_to_tensor(0, k_out, elements)
            buffer.copy_to_tensor(page.v_offset, v_out, elements)
            logical = 2 * elements * k_out.element_size()
            if self.store.kv_layout.storage_dtype == "int8":
                if k_scales_out is None or v_scales_out is None:
                    raise ValueError("INT8 K/V reads require scale buffers")
                groups = math.ceil(
                    page.valid_tokens / self.store.kv_layout.quant_group_tokens
                )
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
        for path, expected in ((self.q_path, q_size), (self.kv_path, kv_size)):
            try:
                actual = path.stat().st_size
            except OSError as error:
                raise ValueError(f"manifest data file is missing: {path}") from error
            if actual != expected:
                raise ValueError(
                    f"manifest size mismatch for {path.name}: {actual} != {expected}"
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
        if _expected_file_size(pages) != file_size:
            raise ValueError(f"{name} page table does not match file size")

    def open_reader(self, memory_plan: HostMemoryPlan, queue_depth: int) -> PageReader:
        return _NvmePageReader(self, memory_plan, queue_depth)


class _NvmeOutputWriter(PageWriter):
    def __init__(
        self,
        sink: "NvmeOutputSink",
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> None:
        self.sink = sink
        self.layout = layout
        self.cu_seqlens = tuple(cu_seqlens)
        self.pages = _q_record_pages(pages, layout)
        self.page_map = {page.page_id: page for page in self.pages}
        self.path = sink.path
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "manifest.json").exists():
            raise FileExistsError(f"output store already exists: {self.path}")
        self.tmp_path = self.path / "out.bin.tmp"
        self.final_path = self.path / "out.bin"
        max_record = max((page.storage_bytes for page in self.pages), default=DIRECT_IO_ALIGNMENT)
        self.buffers = [_AlignedBuffer(max_record) for _ in range(queue_depth)]
        self.registered_bytes = sum(buffer.size_bytes for buffer in self.buffers)
        try:
            memory_plan.register("bounce", self.registered_bytes)
        except BaseException:
            for buffer in self.buffers:
                buffer.close()
            raise
        self.memory_plan = memory_plan
        self.available: queue.LifoQueue[_AlignedBuffer] = queue.LifoQueue()
        for buffer in self.buffers:
            self.available.put(buffer)
        try:
            self.fd = _open_file(
                self.tmp_path,
                os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
                0o644,
                sink.direct_io,
            )
        except BaseException:
            memory_plan.release("bounce", self.registered_bytes)
            for buffer in self.buffers:
                buffer.close()
            raise
        self.closed = False
        self._write_lock = threading.Lock()

    def write_page(self, page: PageDescriptor, data: torch.Tensor) -> PageReadMetrics:
        record = self.page_map[page.page_id]
        expected = (page.valid_tokens, self.layout.heads, self.layout.head_dim)
        if data.device.type != "cpu" or tuple(data[: page.valid_tokens].shape) != expected:
            raise ValueError("output page tensor does not match its descriptor")
        if data.dtype != self.layout.torch_dtype:
            raise ValueError("output page dtype does not match output layout")
        buffer = self.available.get()
        started = time.perf_counter()
        try:
            buffer.zero(record.storage_bytes)
            buffer.copy_tensor(0, data[: page.valid_tokens])
            view = memoryview(buffer.mapping)[: record.storage_bytes]
            try:
                with self._write_lock:
                    written = os.pwritev(self.fd, [view], record.file_offset)
            except OSError as error:
                if self.sink.direct_io and error.errno in {
                    errno.EINVAL,
                    errno.EOPNOTSUPP,
                    errno.ENOTSUP,
                }:
                    raise RuntimeError(
                        f"filesystem rejected required O_DIRECT write for {self.tmp_path}"
                    ) from error
                raise
            finally:
                view.release()
            if written != record.storage_bytes:
                raise OSError(errno.EIO, f"short output write: {written} != {record.storage_bytes}")
        finally:
            self.available.put(buffer)
        logical = page.valid_tokens * self.layout.bytes_per_token
        return PageReadMetrics(
            read_seconds=time.perf_counter() - started,
            logical_bytes=logical,
            physical_bytes=record.storage_bytes,
        )

    def close(self) -> Path:
        if self.closed:
            return self.path
        try:
            try:
                try:
                    os.fsync(self.fd)
                finally:
                    os.close(self.fd)
                if self.tmp_path.stat().st_size != _expected_file_size(self.pages):
                    raise OSError(errno.EIO, "output file size does not match its page table")
                os.replace(self.tmp_path, self.final_path)
                _fsync_directory(self.path)
                manifest = {
                    "format": "seqattn-output",
                    "version": FORMAT_VERSION,
                    "alignment_bytes": DIRECT_IO_ALIGNMENT,
                    "layout": self.layout.as_dict(),
                    "cu_seqlens": list(self.cu_seqlens),
                    "pages": [page.as_dict() for page in self.pages],
                    "files": {
                        "output": {
                            "name": "out.bin",
                            "size_bytes": _expected_file_size(self.pages),
                        }
                    }
                }
                _atomic_write_json(self.path / "manifest.json", manifest)
            except BaseException:
                for path in (self.tmp_path, self.final_path, self.path / "manifest.json.tmp"):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                raise
        finally:
            self.memory_plan.release("bounce", self.registered_bytes)
            for buffer in self.buffers:
                buffer.close()
            self.closed = True
        return self.path

    def abort(self) -> None:
        if self.closed:
            return
        try:
            os.close(self.fd)
        finally:
            self.memory_plan.release("bounce", self.registered_bytes)
            for buffer in self.buffers:
                buffer.close()
            self.closed = True
        for path in (self.tmp_path, self.path / "manifest.json.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class NvmeOutputSink(PageSink):
    backing_kind = "nvme"

    def __init__(self, path: str | os.PathLike[str], *, direct_io: bool = True) -> None:
        self.path = Path(path)
        self.direct_io = direct_io

    def open_writer(
        self,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> PageWriter:
        return _NvmeOutputWriter(
            self,
            layout,
            cu_seqlens,
            pages,
            memory_plan,
            queue_depth,
        )


def load_nvme_output(
    path: str | os.PathLike[str], *, direct_io: bool = False
) -> torch.Tensor:
    directory = Path(path)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read output manifest") from error
    if manifest.get("format") != "seqattn-output" or manifest.get("version") != FORMAT_VERSION:
        raise ValueError("unsupported or corrupt seqattn output manifest")
    layout = TensorLayout(**manifest["layout"])
    pages = tuple(PageDescriptor.from_dict(page) for page in manifest["pages"])
    output = torch.empty(
        (layout.total_tokens, layout.heads, layout.head_dim), dtype=layout.torch_dtype
    )
    file_path = directory / manifest["files"]["output"]["name"]
    expected_size = int(manifest["files"]["output"]["size_bytes"])
    if file_path.stat().st_size != expected_size or _expected_file_size(pages) != expected_size:
        raise ValueError("output file size does not match manifest")
    fd = _open_file(file_path, os.O_RDONLY, 0, direct_io)
    buffer = _AlignedBuffer(
        max((page.storage_bytes for page in pages), default=DIRECT_IO_ALIGNMENT)
    )
    try:
        for page in pages:
            view = memoryview(buffer.mapping)[: page.storage_bytes]
            try:
                read = os.preadv(fd, [view], page.file_offset)
            finally:
                view.release()
            if read != page.storage_bytes:
                raise OSError(errno.EIO, "short output read")
            target = output[page.token_start : page.token_stop]
            buffer.copy_to_tensor(0, target, target.numel())
    finally:
        os.close(fd)
        buffer.close()
    return output


@contextmanager
def ephemeral_nvme_directory(
    parent: str | os.PathLike[str] | None = None,
    *,
    prefix: str = "seqattn-",
) -> Iterator[Path]:
    """Explicitly managed temporary store directory."""

    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "DIRECT_IO_ALIGNMENT",
    "NvmeOutputSink",
    "NvmeQKVStore",
    "NvmeQKVWriter",
    "ephemeral_nvme_directory",
    "load_nvme_output",
]
