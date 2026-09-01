from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    ProjectedCrossAttentionRunner,
    ProjectedCrossAttentionStats,
    ProjectionPipelineConfig,
    RecomputedCrossAttentionRunner,
    RecomputedCrossAttentionStats,
    StreamingAttentionConfig,
    build_plan,
)
from seqattn_core.kernels import triton_is_available


def _segmented_cross_attention(q, k, v, q_bounds, k_bounds, scale):
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


def _cross_modules(*, query_features, context_features, q_heads, kv_heads, head_dim, dtype):
    q_linear = torch.nn.Linear(query_features, q_heads * head_dim, bias=False).to("cuda", dtype)
    kv_linear = torch.nn.Linear(context_features, 2 * kv_heads * head_dim, bias=False).to(
        "cuda", dtype
    )
    out_linear = torch.nn.Linear(q_heads * head_dim, query_features, bias=False).to("cuda", dtype)
    return q_linear, kv_linear, out_linear


def _attention_config(*, output_mode="host"):
    return StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=19,
        block_m=16,
        block_n=16,
        output_mode=output_mode,
    )


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_projected_cross_attention_matches_full_gpu_gqa_and_packed_batch():
    torch.manual_seed(503)
    dtype = torch.bfloat16
    query_tokens = 97
    context_tokens = 41
    query_features = 48
    context_features = 36
    q_heads = 4
    kv_heads = 2
    head_dim = 16
    q_bounds = [0, 37, 37, query_tokens]
    k_bounds = [0, 13, 13, context_tokens]
    cu_q = torch.tensor(q_bounds, dtype=torch.int32)
    cu_k = torch.tensor(k_bounds, dtype=torch.int32)
    query = torch.randn(query_tokens, query_features, dtype=dtype, pin_memory=True)
    context = torch.randn(context_tokens, context_features, dtype=dtype, pin_memory=True)
    q_linear, kv_linear, out_linear = _cross_modules(
        query_features=query_features,
        context_features=context_features,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    def project_q(tile, start, stop):
        del start, stop
        return q_linear(tile).view(-1, q_heads, head_dim)

    def project_kv(tile, start, stop):
        del start, stop
        kv = kv_linear(tile).view(-1, 2, kv_heads, head_dim)
        return kv[:, 0].contiguous(), kv[:, 1].contiguous()

    def output_projector(attention, start, stop):
        del start, stop
        return out_linear(attention)

    config = _attention_config()
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=query_tokens,
        max_kv_tokens=context_tokens,
        config=config,
    )
    runner = ProjectedCrossAttentionRunner(
        plan,
        config,
        ProjectionPipelineConfig(projection_tile_tokens=23),
    )
    stats = ProjectedCrossAttentionStats()
    actual = runner(
        query,
        context,
        cu_q,
        cu_k,
        project_q=project_q,
        project_kv=project_kv,
        output_projector=output_projector,
        output_features=query_features,
        stats=stats,
    )

    q = q_linear(query.to("cuda")).view(-1, q_heads, head_dim)
    kv = kv_linear(context.to("cuda")).view(-1, 2, kv_heads, head_dim)
    expected_attention = _segmented_cross_attention(
        q,
        kv[:, 0],
        kv[:, 1],
        q_bounds,
        k_bounds,
        head_dim**-0.5,
    )
    expected = out_linear(expected_attention.reshape(query_tokens, -1)).cpu()

    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=1e-2)
    assert stats.q_projection_tokens == query_tokens
    assert stats.kv_projection_tokens == context_tokens
    assert (
        stats.qkv_host_bytes
        == (query_tokens * q_heads + 2 * context_tokens * kv_heads)
        * head_dim
        * query.element_size()
    )


class _CaptureConsumer:
    def __init__(self, tokens, features, device, dtype):
        self.output = torch.empty((tokens, features), device=device, dtype=dtype)
        self.task_done = torch.cuda.Event(enable_timing=True)

    def begin_task(self, task):
        del task

    def __call__(self, attention, start, stop):
        self.output[start:stop].copy_(attention)

    def finish(self):
        pass

    def finish_task(self):
        self.task_done.record(torch.cuda.current_stream(self.output.device))
        return self.task_done

    def synchronize(self):
        torch.cuda.synchronize(self.output.device)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_recomputed_cross_attention_matches_materialized_reference_without_host_qkv():
    torch.manual_seed(509)
    dtype = torch.bfloat16
    query_tokens = 73
    context_tokens = 29
    query_features = 40
    context_features = 24
    q_heads = 4
    kv_heads = 2
    head_dim = 16
    q_bounds = [0, 31, query_tokens]
    k_bounds = [0, 11, context_tokens]
    query = torch.randn(query_tokens, query_features, dtype=dtype, pin_memory=True)
    context = torch.randn(context_tokens, context_features, dtype=dtype, pin_memory=True)
    q_linear, kv_linear, _ = _cross_modules(
        query_features=query_features,
        context_features=context_features,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    config = _attention_config(output_mode="device_consumer")
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=query_tokens,
        max_kv_tokens=context_tokens,
        config=config,
    )
    runner = RecomputedCrossAttentionRunner(
        plan,
        query_hidden_features=query_features,
        context_hidden_features=context_features,
        attention_config=config,
    )

    def project_q(tile, destination, start, stop):
        del start, stop
        destination.copy_(q_linear(tile).view(-1, q_heads, head_dim))

    def project_kv(tile, destination_k, destination_v, start, stop):
        del start, stop
        kv = kv_linear(tile).view(-1, 2, kv_heads, head_dim)
        destination_k.copy_(kv[:, 0])
        destination_v.copy_(kv[:, 1])

    consumer = _CaptureConsumer(query_tokens, q_heads * head_dim, "cuda", dtype)
    stats = RecomputedCrossAttentionStats()
    runner.run_with_device_consumer(
        query,
        context,
        torch.tensor(q_bounds, dtype=torch.int32),
        torch.tensor(k_bounds, dtype=torch.int32),
        project_q=project_q,
        project_kv=project_kv,
        output_consumer=consumer,
        stats=stats,
    )
    q = q_linear(query.to("cuda")).view(-1, q_heads, head_dim)
    kv = kv_linear(context.to("cuda")).view(-1, 2, kv_heads, head_dim)
    expected = _segmented_cross_attention(
        q,
        kv[:, 0],
        kv[:, 1],
        q_bounds,
        k_bounds,
        head_dim**-0.5,
    )

    torch.testing.assert_close(consumer.output.reshape_as(expected), expected, atol=7e-2, rtol=1e-2)
    assert stats.qkv_host_bytes == 0
    assert stats.attention.h2d_bytes == 0
    assert runner.workspace.query.hidden.shape[0] == plan.q_chunk_tokens
    assert runner.workspace.context.hidden.shape[0] == plan.kv_chunk_tokens


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
def test_projected_cross_runner_recovers_after_projection_callback_failure():
    dtype = torch.bfloat16
    query_tokens, context_tokens = 31, 23
    query_features, context_features = 24, 20
    q_heads, kv_heads, head_dim = 2, 1, 16
    query = torch.randn(query_tokens, query_features, dtype=dtype, pin_memory=True)
    context = torch.randn(context_tokens, context_features, dtype=dtype, pin_memory=True)
    q_linear, kv_linear, _ = _cross_modules(
        query_features=query_features,
        context_features=context_features,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=16,
        kv_chunk_tokens=16,
        block_m=16,
        block_n=16,
    )
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=query_tokens,
        max_kv_tokens=context_tokens,
        config=config,
    )
    runner = ProjectedCrossAttentionRunner(
        plan,
        config,
        ProjectionPipelineConfig(projection_tile_tokens=11),
    )

    def fail_query(tile, start, stop):
        del tile, start, stop
        raise RuntimeError("injected cross projection failure")

    def project_kv(tile, start, stop):
        del start, stop
        projected = kv_linear(tile).view(-1, 2, kv_heads, head_dim)
        return projected[:, 0].contiguous(), projected[:, 1].contiguous()

    with pytest.raises(RuntimeError, match="injected cross projection failure"):
        runner.project_to_host(
            query,
            context,
            project_q=fail_query,
            project_kv=project_kv,
        )

    def project_q(tile, start, stop):
        del start, stop
        return q_linear(tile).view(-1, q_heads, head_dim)

    q, k, v = runner.project_to_host(
        query,
        context,
        project_q=project_q,
        project_kv=project_kv,
    )
    expected_q = q_linear(query.to("cuda")).view_as(q).cpu()
    expected_kv = kv_linear(context.to("cuda")).view(-1, 2, kv_heads, head_dim).cpu()
    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
    torch.testing.assert_close(k, expected_kv[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(v, expected_kv[:, 1], atol=0, rtol=0)
