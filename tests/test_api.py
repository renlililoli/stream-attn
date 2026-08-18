import pytest
import torch

from seqattn import StreamingAttentionConfig, streaming_attn_func


def test_dense_api_reference_backend():
    torch.manual_seed(11)
    q = torch.randn(2, 7, 4, 32)
    k = torch.randn(2, 9, 2, 32)
    v = torch.randn_like(k)
    output = streaming_attn_func(
        q,
        k,
        v,
        causal=True,
        device="cpu",
        config=StreamingAttentionConfig(
            backend="reference",
            q_chunk_tokens=3,
            kv_chunk_tokens=4,
        ),
    )
    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_dense_api_rejects_dropout():
    q = torch.randn(1, 2, 1, 8)
    with pytest.raises(ValueError, match="dropout_p=0"):
        streaming_attn_func(
            q,
            q,
            q,
            dropout_p=0.1,
            device="cpu",
            config=StreamingAttentionConfig(backend="reference"),
        )
