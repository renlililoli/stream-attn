import pytest
import torch

from seqattn import (
    MemoryPageSink,
    MemoryPageSource,
    PagedAttentionConfig,
    PagedAttentionRunner,
    PagedAttentionStats,
    StreamingAttentionConfig,
)
from seqattn.kernels import triton_is_available
from seqattn.reference import streaming_attention_reference


pytestmark = pytest.mark.skipif(
    not triton_is_available(), reason="requires CUDA and Triton"
)


@pytest.mark.parametrize("storage_dtype", ["bf16", "int8"])
def test_paged_triton_matches_reference(storage_dtype):
    torch.manual_seed(401)
    q = torch.randn(73, 8, 32, dtype=torch.bfloat16)
    k = torch.randn(89, 2, 32, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 31, 31, 73], dtype=torch.int32)
    cu_k = torch.tensor([0, 37, 37, 89], dtype=torch.int32)
    source = MemoryPageSource(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        page_target_bytes=4096,
        block_n=16,
        kv_storage_dtype=storage_dtype,
    )
    config = PagedAttentionConfig(
        attention=StreamingAttentionConfig(
            backend="triton",
            workspace_budget_bytes=128 * 2**20,
            q_chunk_tokens=23,
            kv_chunk_tokens=32,
            block_m=16,
            block_n=16,
            num_output_buffers=2,
        ),
        host_memory_budget_bytes=64 * 2**20,
        pinned_staging_budget_bytes=16 * 2**20,
        direct_io_bounce_budget_bytes=8 * 2**20,
        metadata_margin_bytes=4 * 2**20,
        page_target_bytes=4096,
        io_workers=2,
        io_queue_depth=2,
        num_output_buffers=2,
        direct_io=False,
        kv_storage_dtype=storage_dtype,
    )
    out = torch.empty_like(q)
    stats = PagedAttentionStats()
    PagedAttentionRunner(config, device="cuda").run(
        source,
        source,
        cu_q,
        cu_k,
        MemoryPageSink(out),
        causal=True,
        stats=stats,
    )
    reference = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=23,
        kv_chunk_tokens=32,
        device="cuda",
        causal=True,
    )
    relative_l2 = (out - reference).float().norm() / reference.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        out.flatten().float(), reference.flatten().float(), dim=0
    )
    if storage_dtype == "bf16":
        assert relative_l2 < 0.005
        assert (out - reference).abs().max() <= 0.015625
    else:
        assert relative_l2 < 0.03
        assert cosine > 0.995
    assert stats.operator_host_peak_bytes <= config.host_memory_budget_bytes

