import itertools

import pytest
import torch

from seqattn import (
    CallbackOutputSink,
    MemoryPageSink,
    MemoryPageSource,
    PagedAttentionConfig,
    PagedAttentionRunner,
    PagedAttentionStats,
    StreamingAttentionConfig,
)
from seqattn.reference import streaming_attention_reference


def make_bounds(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


def paged_config(storage_dtype="fp32", *, cache_mib=4):
    return PagedAttentionConfig(
        attention=StreamingAttentionConfig(
            backend="reference",
            q_chunk_tokens=7,
            kv_chunk_tokens=16,
            block_m=16,
            block_n=16,
        ),
        host_memory_budget_bytes=(cache_mib + 3) * 2**20,
        pinned_staging_budget_bytes=1 * 2**20,
        direct_io_bounce_budget_bytes=1 * 2**20,
        metadata_margin_bytes=1 * 2**20,
        page_target_bytes=2048,
        io_workers=2,
        io_queue_depth=2,
        num_output_buffers=2,
        direct_io=False,
        kv_storage_dtype=storage_dtype,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
def test_memory_paged_matches_fp32_reference(causal, q_heads, kv_heads):
    torch.manual_seed(211)
    cu_q = make_bounds([19, 0, 23])
    cu_k = make_bounds([21, 0, 29])
    q = torch.randn(42, q_heads, 32)
    k = torch.randn(50, kv_heads, 32)
    v = torch.randn_like(k)
    source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="fp32",
    )
    out = torch.empty_like(q)
    stats = PagedAttentionStats()
    actual = PagedAttentionRunner(paged_config(), device="cpu").run(
        source,
        source,
        cu_q,
        cu_k,
        MemoryPageSink(out),
        causal=causal,
        stats=stats,
    )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=7,
        kv_chunk_tokens=16,
        causal=causal,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert stats.operator_host_peak_bytes <= stats.host_memory_budget_bytes
    assert stats.q_pages == len(source.q_pages)
    assert stats.cache_hits + stats.cache_misses == stats.kv_pages


def test_pages_never_cross_packed_segment_boundaries():
    q = torch.randn(53, 4, 16)
    k = torch.randn(61, 2, 16)
    v = torch.randn_like(k)
    cu_q = make_bounds([17, 0, 36])
    cu_k = make_bounds([19, 0, 42])
    source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        page_target_bytes=1024,
        block_n=16,
    )
    for pages, bounds in ((source.q_pages, cu_q.tolist()), (source.kv_pages, cu_k.tolist())):
        for page in pages:
            assert bounds[page.segment_id] <= page.token_start
            assert page.token_stop <= bounds[page.segment_id + 1]
            assert page.padded_tokens % 16 == 0
    assert not any(page.segment_id == 1 for page in source.q_pages)
    assert not any(page.segment_id == 1 for page in source.kv_pages)


def test_callback_sink_reuses_bounded_output_ring():
    torch.manual_seed(223)
    q = torch.randn(97, 4, 16)
    k = torch.randn(97, 2, 16)
    v = torch.randn_like(k)
    cu = make_bounds([97])
    source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="fp32",
    )
    pages = []
    storage_pointers = set()

    def consume(page, data):
        pages.append((page, data.clone()))
        storage_pointers.add(data.untyped_storage().data_ptr())

    result = PagedAttentionRunner(paged_config(), device="cpu").run(
        source,
        source,
        cu,
        cu,
        CallbackOutputSink(consume),
    )
    assert result == len(source.q_pages)
    assert len(storage_pointers) <= 2
    assembled = torch.empty_like(q)
    for page, data in pages:
        assembled[page.token_start : page.token_stop].copy_(data)
    expected = streaming_attention_reference(
        q, k, v, cu, cu, q_chunk_tokens=7, kv_chunk_tokens=16
    )
    torch.testing.assert_close(assembled, expected, atol=1e-5, rtol=1e-5)


def test_int8_is_explicit_and_reports_approximation_error():
    torch.manual_seed(227)
    q = torch.randn(71, 4, 16)
    k = torch.randn(79, 2, 16)
    v = torch.randn_like(k)
    cu_q = make_bounds([31, 40])
    cu_k = make_bounds([37, 42])
    source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        page_target_bytes=2048,
        block_n=16,
        kv_storage_dtype="int8",
    )
    out = torch.empty_like(q)
    stats = PagedAttentionStats()
    PagedAttentionRunner(paged_config("int8"), device="cpu").run(
        source,
        source,
        cu_q,
        cu_k,
        MemoryPageSink(out),
        stats=stats,
    )
    exact = streaming_attention_reference(
        q, k, v, cu_q, cu_k, q_chunk_tokens=7, kv_chunk_tokens=16
    )
    relative_l2 = (out - exact).float().norm() / exact.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten().float(), exact.flatten().float(), dim=0
    )
    assert 0 < relative_l2 < 0.05
    assert cosine > 0.995
    assert stats.kv_storage_dtype == "int8"
    assert stats.quantization_seconds > 0


def test_storage_dtype_mismatch_is_rejected():
    q = torch.randn(17, 1, 16)
    cu = make_bounds([17])
    source = MemoryPageSource(
        q=q,
        k=q,
        v=q,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        kv_storage_dtype="int8",
        block_n=16,
    )
    with pytest.raises(ValueError, match="storage dtype"):
        PagedAttentionRunner(paged_config("fp32"), device="cpu").run(
            source, source, cu, cu, MemoryPageSink(torch.empty_like(q))
        )

