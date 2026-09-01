from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    ProjectedAttentionRunner,
    ProjectedCrossAttentionRunner,
    ProjectionPipelineConfig,
    RecomputedAttentionRunner,
    RecomputedCrossAttentionRunner,
    StreamingAttentionConfig,
    build_plan,
)
from seqattn_core.dit.wan import (
    WanBlockOps,
    WanMaterializedProjections,
    WanMaterializedRunner,
    WanRecomputeProjections,
    WanRecomputeRunner,
    WanSequenceMeta,
)
from seqattn_core.kernels import triton_is_available
from seqattn_core.projection import (
    CrossProjection,
    CrossRecomputeProjection,
    SelfProjection,
    SelfRecomputeProjection,
)


def _segmented_attention(q, k, v, q_bounds, k_bounds, scale):
    output = torch.empty_like(q)
    repeat = q.shape[1] // k.shape[1]
    for (q_start, q_stop), (k_start, k_stop) in zip(pairwise(q_bounds), pairwise(k_bounds)):
        if q_start == q_stop:
            continue
        expanded_k = k[k_start:k_stop].repeat_interleave(repeat, dim=1)
        expanded_v = v[k_start:k_stop].repeat_interleave(repeat, dim=1)
        tile = torch.nn.functional.scaled_dot_product_attention(
            q[q_start:q_stop].transpose(0, 1).unsqueeze(0),
            expanded_k.transpose(0, 1).unsqueeze(0),
            expanded_v.transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
        output[q_start:q_stop].copy_(tile.squeeze(0).transpose(0, 1))
    return output


def _build_modules(hidden_features, text_features, heads, cross_kv_heads, head_dim, dtype):
    inner = heads * head_dim
    self_qkv = torch.nn.Linear(hidden_features, 3 * inner, bias=False).to("cuda", dtype)
    self_out = torch.nn.Linear(inner, hidden_features, bias=False).to("cuda", dtype)
    cross_q = torch.nn.Linear(hidden_features, inner, bias=False).to("cuda", dtype)
    cross_kv = torch.nn.Linear(text_features, 2 * cross_kv_heads * head_dim, bias=False).to(
        "cuda", dtype
    )
    cross_out = torch.nn.Linear(inner, hidden_features, bias=False).to("cuda", dtype)
    ffn_in = torch.nn.Linear(hidden_features, 2 * 80, bias=False).to("cuda", dtype)
    ffn_out = torch.nn.Linear(80, hidden_features, bias=False).to("cuda", dtype)
    return self_qkv, self_out, cross_q, cross_kv, cross_out, ffn_in, ffn_out


def _reference_block(hidden, text, modules, hidden_bounds, text_bounds, heads, kv_heads, head_dim):
    self_qkv, self_out, cross_q, cross_kv, cross_out, ffn_in, ffn_out = modules
    tokens = hidden.shape[0]
    self_projected = self_qkv(hidden).view(tokens, 3, heads, head_dim)
    self_attention = _segmented_attention(
        self_projected[:, 0],
        self_projected[:, 1],
        self_projected[:, 2],
        hidden_bounds,
        hidden_bounds,
        head_dim**-0.5,
    )
    post_self = hidden + self_out(self_attention.reshape(tokens, -1))
    q = cross_q(post_self).view(tokens, heads, head_dim)
    kv = cross_kv(text).view(text.shape[0], 2, kv_heads, head_dim)
    cross_attention = _segmented_attention(
        q,
        kv[:, 0],
        kv[:, 1],
        hidden_bounds,
        text_bounds,
        head_dim**-0.5,
    )
    post_cross = post_self + cross_out(cross_attention.reshape(tokens, -1))
    gate, up = ffn_in(post_cross).chunk(2, dim=-1)
    return post_cross + ffn_out(torch.nn.functional.silu(gate) * up)


def _plans(tokens, text_tokens, heads, kv_heads, head_dim, dtype):
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=19,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    self_plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    cross_plan = build_plan(
        q_heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=text_tokens,
        config=config,
    )
    return config, self_plan, cross_plan


def _ops(modules, device):
    _, self_out, _, _, cross_out, ffn_in, ffn_out = modules

    def self_epilogue(attention, residual_host, start, stop):
        return self_out(attention) + residual_host[start:stop].to(device, non_blocking=True)

    def cross_epilogue(attention, residual_host, start, stop):
        return cross_out(attention) + residual_host[start:stop].to(device, non_blocking=True)

    def ffn(tile, start, stop):
        del start, stop
        gate, up = ffn_in(tile).chunk(2, dim=-1)
        return tile + ffn_out(torch.nn.functional.silu(gate) * up)

    return WanBlockOps(self_epilogue, cross_epilogue, ffn)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_wan_materialized_block_matches_full_gpu_order():
    torch.manual_seed(601)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokens = 97
    text_tokens = 41
    hidden_features = 48
    text_features = 32
    heads = 4
    kv_heads = 2
    head_dim = 16
    hidden_bounds = [0, 37, tokens]
    text_bounds = [0, 13, text_tokens]
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = hidden.clone()
    text = torch.randn(text_tokens, text_features, dtype=dtype, pin_memory=True)
    modules = _build_modules(hidden_features, text_features, heads, kv_heads, head_dim, dtype)
    self_qkv, _, cross_q, cross_kv, *_ = modules
    config, self_plan, cross_plan = _plans(tokens, text_tokens, heads, kv_heads, head_dim, dtype)
    runner = WanMaterializedRunner(
        ProjectedAttentionRunner(
            self_plan,
            config,
            ProjectionPipelineConfig(projection_chunk_tokens=23),
        ),
        ProjectedCrossAttentionRunner(
            cross_plan,
            config,
            ProjectionPipelineConfig(projection_chunk_tokens=17),
        ),
        hidden_features=hidden_features,
        ffn_chunk_tokens=29,
    )

    def project_self(tile, start, stop):
        del start, stop
        qkv = self_qkv(tile).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def project_cross_q(tile, start, stop):
        del start, stop
        return cross_q(tile).view(-1, heads, head_dim)

    def project_cross_kv(tile, start, stop):
        del start, stop
        kv = cross_kv(tile).view(-1, 2, kv_heads, head_dim)
        return kv[:, 0].contiguous(), kv[:, 1].contiguous()

    runner.run_block_(
        hidden,
        text,
        WanSequenceMeta(
            torch.tensor(hidden_bounds, dtype=torch.int32),
            torch.tensor(text_bounds, dtype=torch.int32),
        ),
        WanMaterializedProjections(
            SelfProjection(project_self),
            CrossProjection(project_cross_q, project_cross_kv),
        ),
        _ops(modules, device),
    )
    expected = _reference_block(
        original.to(device),
        text.to(device),
        modules,
        hidden_bounds,
        text_bounds,
        heads,
        kv_heads,
        head_dim,
    )

    torch.testing.assert_close(hidden, expected.cpu(), atol=8e-2, rtol=1e-2)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_wan_recompute_block_matches_full_gpu_without_host_qkv():
    torch.manual_seed(607)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokens = 89
    text_tokens = 43
    hidden_features = 48
    text_features = 32
    heads = 4
    kv_heads = 2
    head_dim = 16
    hidden_bounds = [0, 31, tokens]
    text_bounds = [0, 17, text_tokens]
    source = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = source.clone()
    destination = torch.empty_like(source, pin_memory=True)
    text = torch.randn(text_tokens, text_features, dtype=dtype, pin_memory=True)
    modules = _build_modules(hidden_features, text_features, heads, kv_heads, head_dim, dtype)
    self_qkv, _, cross_q, cross_kv, *_ = modules
    config, self_plan, cross_plan = _plans(tokens, text_tokens, heads, kv_heads, head_dim, dtype)
    self_runner = RecomputedAttentionRunner(
        self_plan,
        hidden_features=hidden_features,
        attention_config=config,
    )
    cross_runner = RecomputedCrossAttentionRunner(
        cross_plan,
        query_hidden_features=hidden_features,
        context_hidden_features=text_features,
        attention_config=config,
    )
    runner = WanRecomputeRunner(self_runner, cross_runner, ffn_chunk_tokens=23)

    def self_q(tile, out, start, stop):
        del start, stop
        out.copy_(self_qkv(tile)[:, : heads * head_dim].view_as(out))

    def self_kv(tile, out_k, out_v, start, stop):
        del start, stop
        qkv = self_qkv(tile).view(-1, 3, heads, head_dim)
        out_k.copy_(qkv[:, 1])
        out_v.copy_(qkv[:, 2])

    def text_q(tile, out, start, stop):
        del start, stop
        out.copy_(cross_q(tile).view_as(out))

    def text_kv(tile, out_k, out_v, start, stop):
        del start, stop
        kv = cross_kv(tile).view(-1, 2, kv_heads, head_dim)
        out_k.copy_(kv[:, 0])
        out_v.copy_(kv[:, 1])

    result = runner.run_block(
        source,
        destination,
        text,
        WanSequenceMeta(
            torch.tensor(hidden_bounds, dtype=torch.int32),
            torch.tensor(text_bounds, dtype=torch.int32),
        ),
        WanRecomputeProjections(
            SelfRecomputeProjection(self_q, self_kv),
            CrossRecomputeProjection(text_q, text_kv),
        ),
        _ops(modules, device),
    )
    expected = _reference_block(
        original.to(device),
        text.to(device),
        modules,
        hidden_bounds,
        text_bounds,
        heads,
        kv_heads,
        head_dim,
    )

    assert result.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(source, original, atol=0, rtol=0)
    torch.testing.assert_close(destination, expected.cpu(), atol=8e-2, rtol=1e-2)
    assert not any(hasattr(self_runner, name) for name in ("q_cpu", "k_cpu", "v_cpu"))
    assert not any(hasattr(cross_runner, name) for name in ("q_cpu", "k_cpu", "v_cpu"))
