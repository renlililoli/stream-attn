import itertools

import pytest
import torch

from seqattn_core.reference import streaming_attention_reference


def make_bounds(lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


def dense_reference(q, k, v, cu_q, cu_k, scale, causal):
    output = torch.empty_like(q)
    group_size = q.shape[1] // k.shape[1]
    for qs, qe, ks, ke in zip(
        cu_q[:-1].tolist(),
        cu_q[1:].tolist(),
        cu_k[:-1].tolist(),
        cu_k[1:].tolist(),
    ):
        if qs == qe:
            continue
        q_tile = q[qs:qe].transpose(0, 1).float()
        k_tile = k[ks:ke].repeat_interleave(group_size, dim=1).transpose(0, 1).float()
        v_tile = v[ks:ke].repeat_interleave(group_size, dim=1).transpose(0, 1).float()
        scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
        if causal:
            q_pos = torch.arange(qe - qs)
            k_pos = torch.arange(ke - ks)
            shift = (ke - ks) - (qe - qs)
            scores.masked_fill_(
                ~(k_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + shift).unsqueeze(0),
                -torch.inf,
            )
        probabilities = torch.softmax(scores, dim=-1)
        output[qs:qe] = torch.matmul(probabilities, v_tile).transpose(0, 1).to(q.dtype)
    return output


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("q_chunk,kv_chunk", [(3, 2), (8, 5)])
def test_reference_matches_dense_attention(causal, q_heads, kv_heads, q_chunk, kv_chunk):
    torch.manual_seed(7)
    cu_q = make_bounds([7, 0, 10])
    cu_k = make_bounds([9, 0, 12])
    q = torch.randn(17, q_heads, 32, dtype=torch.float32)
    k = torch.randn(21, kv_heads, 32, dtype=torch.float32)
    v = torch.randn_like(k)
    scale = 0.19
    expected = dense_reference(q, k, v, cu_q, cu_k, scale, causal)
    actual = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=q_chunk,
        kv_chunk_tokens=kv_chunk,
        softmax_scale=scale,
        causal=causal,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_reference_does_not_attend_across_packed_segments():
    q = torch.ones(4, 1, 8)
    k = torch.ones_like(q)
    v = torch.cat((torch.zeros(2, 1, 8), torch.full((2, 1, 8), 10.0)))
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    output = streaming_attention_reference(
        q,
        k,
        v,
        cu,
        cu,
        q_chunk_tokens=1,
        kv_chunk_tokens=1,
    )
    torch.testing.assert_close(output[:2], torch.zeros_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], 10.0))


def test_reference_rejects_query_sequence_without_keys():
    q = torch.randn(2, 1, 8)
    k = torch.empty(0, 1, 8)
    with pytest.raises(ValueError, match="queries but no keys"):
        streaming_attention_reference(
            q,
            k,
            k,
            torch.tensor([0, 2]),
            torch.tensor([0, 0]),
            q_chunk_tokens=2,
            kv_chunk_tokens=2,
        )
