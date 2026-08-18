from __future__ import annotations

import errno
import json
import os
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from ..paged.layout import PageDescriptor, PageReadMetrics, TensorLayout
from ..paged.memory_budget import HostMemoryPlan
from ..paged.protocols import PageSink, PageWriter
from .direct_io import (
    DIRECT_IO_ALIGNMENT,
    FORMAT_VERSION,
    AlignedBuffer,
    _atomic_write_json,
    _fsync_directory,
    _open_file,
)
from .records import expected_file_size, q_record_pages


class _NvmeOutputWriter(PageWriter):
    def __init__(
        self,
        sink: NvmeOutputSink,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> None:
        self.sink = sink
        self.layout = layout
        self.cu_seqlens = tuple(cu_seqlens)
        self.pages = q_record_pages(pages, layout)
        self.page_map = {page.page_id: page for page in self.pages}
        self.path = sink.path
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "manifest.json").exists():
            raise FileExistsError(f"output store already exists: {self.path}")
        self.tmp_path = self.path / "out.bin.tmp"
        self.final_path = self.path / "out.bin"
        max_record = max((page.storage_bytes for page in self.pages), default=DIRECT_IO_ALIGNMENT)
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
                if self.tmp_path.stat().st_size != expected_file_size(self.pages):
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
                            "size_bytes": expected_file_size(self.pages),
                        }
                    },
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


def load_nvme_output(path: str | os.PathLike[str], *, direct_io: bool = False) -> torch.Tensor:
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
    if file_path.stat().st_size != expected_size or expected_file_size(pages) != expected_size:
        raise ValueError("output file size does not match manifest")
    fd = _open_file(file_path, os.O_RDONLY, 0, direct_io)
    buffer = AlignedBuffer(max((page.storage_bytes for page in pages), default=DIRECT_IO_ALIGNMENT))
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


__all__ = ["NvmeOutputSink", "ephemeral_nvme_directory", "load_nvme_output"]
