from __future__ import annotations

import errno
import json
import mmap
import os
from pathlib import Path

import torch

from ..paged.layout import align_up

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


class AlignedBuffer:
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


class AlignedFileWriter:
    def __init__(self, path: Path, max_record_bytes: int, direct_io: bool) -> None:
        self.path = path
        self.direct_io = direct_io
        self.fd = _open_file(
            path,
            os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
            0o644,
            direct_io,
        )
        self.buffer = AlignedBuffer(max_record_bytes)
        self.closed = False

    def write(self, offset: int, storage_bytes: int) -> None:
        if offset % DIRECT_IO_ALIGNMENT or storage_bytes % DIRECT_IO_ALIGNMENT:
            raise ValueError("direct-I/O writes must use 4096-byte aligned offsets and lengths")
        view = memoryview(self.buffer.mapping)[:storage_bytes]
        try:
            written = os.pwritev(self.fd, [view], offset)
        except OSError as error:
            if self.direct_io and error.errno in {
                errno.EINVAL,
                errno.EOPNOTSUPP,
                errno.ENOTSUP,
            }:
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


# Private aliases retained for tests and compatibility with the previous module.
_AlignedBuffer = AlignedBuffer
_AlignedFileWriter = AlignedFileWriter


__all__ = [
    "DIRECT_IO_ALIGNMENT",
    "FORMAT_VERSION",
    "AlignedBuffer",
    "AlignedFileWriter",
]
