import math

import pytest
import torch

from seqattn_core import (
    ProjectedAttentionRunner,
    ProjectedAttentionStats,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)
from seqattn_core.kernels import triton_is_available


def test_projection_pipeline_config_validation():
    with pytest.raises(ValueError, match="projection_chunk_tokens"):
        ProjectionPipelineConfig(projection_chunk_tokens=0).validate()
    with pytest.raises(ValueError, match="num_projection_buffers"):
        ProjectionPipelineConfig(num_projection_buffers=4).validate()


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_projected_pipeline_matches_full_gpu(dtype):
    torch.manual_seed(101)
    tokens = 97
    hidden_features = 48
    heads = 4
    head_dim = 16
    inner = heads * head_dim
    hidden_cpu = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    cu = torch.tensor([0, 37, 37, 97], dtype=torch.int32)
    qkv_linear = torch.nn.Linear(hidden_features, inner * 3, bias=False).to(
        device="cuda", dtype=dtype
    )
    out_linear = torch.nn.Linear(inner, hidden_features, bias=False).to(device="cuda", dtype=dtype)

    def project_qkv(hidden, start, stop):
        del start, stop
        qkv = qkv_linear(hidden).view(-1, heads, 3, head_dim)
        return (
            qkv[:, :, 0, :].contiguous(),
            qkv[:, :, 1, :].contiguous(),
            qkv[:, :, 2, :].contiguous(),
        )

    def output_projector(attention, start, stop):
        del start, stop
        return out_linear(attention)

    attention_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=41,
        kv_chunk_tokens=29,
        block_m=16,
        block_n=16,
        num_output_buffers=2,
    )
    pipeline_config = ProjectionPipelineConfig(
        projection_chunk_tokens=31,
        num_projection_buffers=2,
    )
    plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    runner = ProjectedAttentionRunner(plan, attention_config, pipeline_config)
    stats = ProjectedAttentionStats()
    actual = runner(
        hidden_cpu,
        cu,
        project_qkv=project_qkv,
        output_projector=output_projector,
        output_features=hidden_features,
        stats=stats,
    )

    hidden_gpu = hidden_cpu.to("cuda")
    qkv = qkv_linear(hidden_gpu).view(tokens, heads, 3, head_dim)
    q = qkv[:, :, 0, :]
    k = qkv[:, :, 1, :]
    v = qkv[:, :, 2, :]
    expected_attention = torch.empty_like(q)
    bounds = cu.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start == stop:
            continue
        tile = torch.nn.functional.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=head_dim**-0.5,
        )
        expected_attention[start:stop].copy_(tile.squeeze(0).transpose(0, 1))
    expected = out_linear(expected_attention.reshape(tokens, inner)).cpu()

    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=1e-2)
    assert stats.projection_chunks == math.ceil(tokens / 31)
    assert stats.attention.d2h_bytes == tokens * hidden_features * actual.element_size()
    assert stats.raw_attention_roundtrip_bytes_avoided == (
        2 * tokens * inner * actual.element_size()
    )
    assert stats.projection_qkv_d2h_bytes == 3 * tokens * inner * actual.element_size()


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
def test_projected_runner_reuse_has_bounded_allocator_growth():
    torch.manual_seed(103)
    dtype = torch.bfloat16
    tokens = 257
    hidden_features = 64
    heads = 4
    head_dim = 16
    inner = heads * head_dim
    hidden_cpu = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    qkv_linear = torch.nn.Linear(hidden_features, inner * 3, bias=False).to(
        device="cuda", dtype=dtype
    )
    out_linear = torch.nn.Linear(inner, hidden_features, bias=False).to(device="cuda", dtype=dtype)

    def project_qkv(hidden, start, stop):
        del start, stop
        qkv = qkv_linear(hidden).view(-1, heads, 3, head_dim)
        return tuple(qkv[:, :, index, :].contiguous() for index in range(3))

    def output_projector(attention, start, stop):
        del start, stop
        return out_linear(attention)

    attention_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=128,
        kv_chunk_tokens=96,
        block_m=32,
        block_n=32,
        num_output_buffers=2,
    )
    plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    runner = ProjectedAttentionRunner(
        plan,
        attention_config,
        ProjectionPipelineConfig(projection_chunk_tokens=96),
    )
    out = torch.empty((tokens, hidden_features), dtype=dtype, pin_memory=True)
    runner(
        hidden_cpu,
        cu,
        project_qkv=project_qkv,
        output_projector=output_projector,
        out=out,
    )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(10):
        runner(
            hidden_cpu,
            cu,
            project_qkv=project_qkv,
            output_projector=output_projector,
            out=out,
        )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline + 4 * 2**20
