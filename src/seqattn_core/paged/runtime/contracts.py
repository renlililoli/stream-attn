from __future__ import annotations

from itertools import pairwise

import torch

from ...config import PagedAttentionConfig
from ..layout import KVLayout, TensorLayout, validate_cu_seqlens
from ..protocols import PageSource


def backing_is_nvme(source: PageSource) -> bool:
    return getattr(source, "backing_kind", "memory") == "nvme"


def validate_source_contract(
    q_source: PageSource,
    kv_source: PageSource,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    config: PagedAttentionConfig,
) -> tuple[TensorLayout, KVLayout, list[int], list[int]]:
    if q_source.q_layout is None:
        raise ValueError("q_source does not provide query pages")
    if kv_source.kv_layout is None:
        raise ValueError("kv_source does not provide K/V pages")
    q_layout = q_source.q_layout
    kv_layout = kv_source.kv_layout
    q_bounds = validate_cu_seqlens(cu_seqlens_q, q_layout.total_tokens, "cu_seqlens_q")
    k_bounds = validate_cu_seqlens(cu_seqlens_k, kv_layout.total_tokens, "cu_seqlens_k")
    if len(q_bounds) != len(k_bounds):
        raise ValueError("query and K/V sources must describe the same batch size")
    if q_source.cu_seqlens_q is not None and tuple(q_bounds) != q_source.cu_seqlens_q:
        raise ValueError("cu_seqlens_q does not match q_source")
    if kv_source.cu_seqlens_k is not None and tuple(k_bounds) != kv_source.cu_seqlens_k:
        raise ValueError("cu_seqlens_k does not match kv_source")
    if q_layout.head_dim != kv_layout.head_dim or q_layout.heads % kv_layout.heads:
        raise ValueError("query and K/V head layouts are incompatible")
    if q_layout.dtype != kv_layout.source_dtype:
        raise ValueError("query and K/V source dtype must match")
    if kv_layout.storage_dtype != config.kv_storage_dtype:
        raise ValueError(
            "kv_source storage dtype does not match PagedAttentionConfig: "
            f"{kv_layout.storage_dtype} != {config.kv_storage_dtype}"
        )
    for index, ((q_start, q_stop), (k_start, k_stop)) in enumerate(
        zip(pairwise(q_bounds), pairwise(k_bounds))
    ):
        if q_stop > q_start and k_stop == k_start:
            raise ValueError(f"sequence {index} has queries but no keys")
    for source in {id(q_source): q_source, id(kv_source): kv_source}.values():
        if backing_is_nvme(source) and source.direct_io != config.direct_io:
            raise ValueError(
                "NVMe source direct_io mode must match PagedAttentionConfig; "
                "buffered fallback is never implicit"
            )
    return q_layout, kv_layout, q_bounds, k_bounds


__all__ = ["backing_is_nvme", "validate_source_contract"]
