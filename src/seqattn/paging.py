"""Compatibility facade for page models and in-memory backends."""

from seqattn_core.paged.layout import (
    KVLayout,
    PageDescriptor,
    PageReadMetrics,
    TensorLayout,
    align_down,
    align_up,
    build_page_descriptors,
    dtype_name,
    pages_by_segment,
    replace_page,
    torch_dtype,
    validate_cu_seqlens,
)
from seqattn_core.paged.memory import CallbackOutputSink, MemoryPageSink, MemoryPageSource
from seqattn_core.paged.protocols import PageReader, PageSink, PageSource, PageWriter

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
