import math
from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    H3BlockOps,
    H3DiTRunner,
    H3DiTStats,
    H3SequenceMeta,
    H3TileConfig,
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
    load_h3_tile_config,
)
from seqattn_core.dit import estimate_h3_aux_workspace_bytes
from seqattn_core.kernels import triton_is_available


def test_h3_aux_workspace_estimate_and_sequence_validation():
    expected = (2 * 31 * 48 + 3 * 43 * 48) * 2
    assert estimate_h3_aux_workspace_bytes(
        hidden_features=48,
        dtype=torch.bfloat16,
        projection_chunk_tokens=31,
        num_projection_buffers=2,
        mlp_chunk_tokens=43,
    ) == expected

    meta = H3SequenceMeta(torch.tensor([0, 3, 7], dtype=torch.int32))
    meta.validate(7)
    with pytest.raises(ValueError, match="span"):
        meta.validate(8)
    with pytest.raises(ValueError, match="int32"):
        H3SequenceMeta(torch.tensor([0, 7], dtype=torch.int64)).validate(7)


def test_h3_tile_config_from_toml(tmp_path):
    path = tmp_path / "seqattn.toml"
    path.write_text(
        "[attention]\nbackend = 'auto'\n\n"
        "[minimax_h3]\nqkv_tile_tokens = 1024\nmlp_tile_tokens = 512\n"
    )
    assert load_h3_tile_config(path) == H3TileConfig(
        qkv_tile_tokens=1024,
        mlp_tile_tokens=512,
    )

    path.write_text("[minimax_h3]\nqkv_tile_tokens = 0\n")
    with pytest.raises(ValueError, match="qkv_tile_tokens"):
        load_h3_tile_config(path)


def _segmented_attention(q, k, v, bounds, scale):
    output = torch.empty_like(q)
    for start, stop in pairwise(bounds):
        if start == stop:
            continue
        tile = torch.nn.functional.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
        output[start:stop].copy_(tile.squeeze(0).transpose(0, 1))
    return output


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_h3_fused_block_reblocks_across_q_ranges_and_matches_full_gpu():
    torch.manual_seed(211)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens = 97
    hidden_features = 48
    heads = 4
    head_dim = 16
    inner = heads * head_dim
    mlp_features = 80
    bounds = [0, 37, 37, tokens]
    cu = torch.tensor(bounds, dtype=torch.int32)

    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = hidden.clone()
    qkv_linear = torch.nn.Linear(hidden_features, inner * 3, bias=False).to(
        device=device, dtype=dtype
    )
    out_linear = torch.nn.Linear(inner, hidden_features, bias=False).to(
        device=device, dtype=dtype
    )
    fc1 = torch.nn.Linear(hidden_features, mlp_features * 2, bias=False).to(
        device=device, dtype=dtype
    )
    fc2 = torch.nn.Linear(mlp_features, hidden_features, bias=False).to(
        device=device, dtype=dtype
    )

    attention_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=29,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    projection_config = ProjectionPipelineConfig(
        projection_chunk_tokens=31,
        num_projection_buffers=2,
    )
    plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    projected = ProjectedAttentionRunner(plan, attention_config, projection_config)
    runner = H3DiTRunner(
        projected,
        hidden_features=hidden_features,
        mlp_chunk_tokens=43,
    )

    def project_qkv(tile, start, stop):
        del start, stop
        qkv = qkv_linear(tile).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def attention_epilogue(attention, start, stop):
        residual = hidden[start:stop].to(device, non_blocking=True)
        return out_linear(attention).add_(residual)

    def mlp(post_attention, start, stop):
        del start, stop
        gate, up = fc1(post_attention).chunk(2, dim=-1)
        update = fc2(torch.nn.functional.silu(gate).mul_(up))
        return post_attention.add_(update)

    stats = H3DiTStats()
    actual_ptr = hidden.data_ptr()
    result = runner.run_block_(
        hidden,
        H3SequenceMeta(cu),
        H3BlockOps(project_qkv, attention_epilogue, mlp),
        softmax_scale=head_dim**-0.5,
        stats=stats,
    )
    assert result.data_ptr() == actual_ptr

    hidden_gpu = original.to(device)
    qkv = qkv_linear(hidden_gpu).view(tokens, 3, heads, head_dim)
    q, k, v = (qkv[:, index] for index in range(3))
    attention = _segmented_attention(q, k, v, bounds, head_dim**-0.5)
    expected = out_linear(attention.reshape(tokens, inner)).add_(hidden_gpu)
    gate, up = fc1(expected).chunk(2, dim=-1)
    expected = expected + fc2(torch.nn.functional.silu(gate) * up)

    torch.testing.assert_close(hidden, expected.cpu(), atol=7e-2, rtol=1e-2)
    assert stats.blocks == 1
    assert stats.mlp_chunks == math.ceil(tokens / 43)
    assert stats.mlp_cross_q_boundaries >= 1
    assert stats.final_hidden_d2h_bytes == hidden.numel() * hidden.element_size()
    assert stats.post_attention_roundtrip_bytes_avoided == (
        2 * hidden.numel() * hidden.element_size()
    )
    assert stats.projection.attention.d2h_bytes == 0
    assert stats.qkv_host_bytes_peak == 3 * tokens * inner * hidden.element_size()
