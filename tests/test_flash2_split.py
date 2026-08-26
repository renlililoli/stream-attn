import pytest
import torch

from seqattn_core import StreamingAttentionConfig, StreamingAttentionRunner, build_plan
from seqattn_core.reference import streaming_attention_reference

try:
    from flash_attn.flash_attn_interface import flash_attn_gpu
except ImportError:
    flash_attn_gpu = None

pytestmark = pytest.mark.skipif(
    flash_attn_gpu is None or not torch.cuda.is_available(),
    reason="requires CUDA and flash-attn",
)


def test_flash2_split_matches_reference():
    torch.manual_seed(43)
    q = torch.randn(257, 4, 64, dtype=torch.bfloat16, pin_memory=True)
    k = torch.randn(319, 2, 64, dtype=torch.bfloat16, pin_memory=True)
    v = torch.randn_like(k).pin_memory()
    cu_q = torch.tensor([0, 257], dtype=torch.int32)
    cu_k = torch.tensor([0, 319], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="fa2",
        q_chunk_tokens=128,
        kv_chunk_tokens=96,
        block_m=64,
        block_n=64,
    )
    plan = build_plan(
        q_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=q.dtype,
        device="cuda",
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        config=config,
    )
    actual = StreamingAttentionRunner(plan, config)(q, k, v, cu_q, cu_k)
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=128,
        kv_chunk_tokens=96,
        device="cuda",
    )
    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=1e-2)


def test_flash2_split_rejects_causal_external_chunks():
    q = torch.randn(64, 2, 64, dtype=torch.bfloat16, pin_memory=True)
    cu = torch.tensor([0, 64], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="fa2",
        q_chunk_tokens=64,
        kv_chunk_tokens=32,
        block_m=64,
        block_n=64,
    )
    plan = build_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=64,
        dtype=q.dtype,
        device="cuda",
        max_q_tokens=64,
        max_kv_tokens=64,
        config=config,
    )
    runner = StreamingAttentionRunner(plan, config)
    with pytest.raises(ValueError, match="causal offsets"):
        runner(q, q, q, cu, cu, causal=True)
