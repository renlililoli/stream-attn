from __future__ import annotations

import errno
import math
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from ..paged.layout import KVLayout, PageDescriptor, TensorLayout, build_page_descriptors
from ..paged.memory import MemoryPageSource
from ..quantization import quantize_int8_per_token_group
from .direct_io import (
    DIRECT_IO_ALIGNMENT,
    FORMAT_VERSION,
    AlignedFileWriter,
    _atomic_write_json,
    _fsync_directory,
)
from .qkv_store import NvmeQKVStore
from .records import expected_file_size, kv_record_pages, q_record_pages


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
        self.q_pages = q_record_pages(q_base, q_layout)
        self.kv_pages = kv_record_pages(kv_base, kv_layout)

    def _validate_q_page(self, page: PageDescriptor, tensor: torch.Tensor) -> None:
        expected = (page.valid_tokens, self.q_layout.heads, self.q_layout.head_dim)
        if tensor.device.type != "cpu" or tuple(tensor.shape) != expected:
            raise ValueError(f"query page {page.page_id} must use CPU shape {expected}")
        if tensor.dtype != self.q_layout.torch_dtype:
            raise ValueError("query page dtype does not match q_layout")

    def _validate_kv_page(self, page: PageDescriptor, k: torch.Tensor, v: torch.Tensor) -> None:
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
    ) -> NvmeQKVStore:
        q_iterator = iter(q_pages)
        kv_iterator = iter(kv_pages)
        q_max = max((page.storage_bytes for page in self.q_pages), default=DIRECT_IO_ALIGNMENT)
        kv_max = max((page.storage_bytes for page in self.kv_pages), default=DIRECT_IO_ALIGNMENT)
        q_writer = None
        kv_writer = None
        try:
            q_writer = AlignedFileWriter(self.q_tmp, q_max, self.direct_io)
            kv_writer = AlignedFileWriter(self.kv_tmp, kv_max, self.direct_io)
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
            if self.q_tmp.stat().st_size != expected_file_size(self.q_pages):
                raise OSError(errno.EIO, "query file size does not match its page table")
            if self.kv_tmp.stat().st_size != expected_file_size(self.kv_pages):
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
                    "q": {"name": "q.bin", "size_bytes": expected_file_size(self.q_pages)},
                    "kv": {"name": "kv.bin", "size_bytes": expected_file_size(self.kv_pages)},
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
    ) -> NvmeQKVStore:
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


__all__ = ["NvmeQKVWriter"]
