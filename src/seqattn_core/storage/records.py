from __future__ import annotations

import math
from collections.abc import Sequence

from ..paged.layout import KVLayout, PageDescriptor, TensorLayout, align_up, replace_page
from .direct_io import DIRECT_IO_ALIGNMENT


def q_record_pages(
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


def kv_record_pages(
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


def expected_file_size(pages: Sequence[PageDescriptor]) -> int:
    if not pages:
        return 0
    last = pages[-1]
    return last.file_offset + last.storage_bytes


__all__ = ["expected_file_size", "kv_record_pages", "q_record_pages"]
