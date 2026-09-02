import math
from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    ProjectedAttentionRunner,
    ProjectedAttentionStats,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_attention_plan,
)
from seqattn_core.kernels import triton_is_available
from seqattn_core.reference import streaming_attention_reference


def test_projection_pipeline_config_validation():
    with pytest.raises(ValueError, match="projection_tile_tokens"):
        ProjectionPipelineConfig(projection_tile_tokens=0).validate()
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
        projection_tile_tokens=31,
        num_projection_buffers=2,
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    runner = ProjectedAttentionRunner(plan, pipeline_config)
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
    for start, stop in pairwise(bounds):
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
    assert stats.projection_tokens == tokens
    assert stats.attention.d2h_bytes == tokens * hidden_features * actual.element_size()
    assert stats.raw_attention_roundtrip_bytes_avoided == (
        2 * tokens * inner * actual.element_size()
    )
    assert stats.projection_qkv_d2h_bytes == 3 * tokens * inner * actual.element_size()


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
def test_projected_runner_recovers_after_projection_callback_failure():
    dtype = torch.bfloat16
    tokens = 37
    hidden_features = 32
    heads = 2
    head_dim = 16
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    linear = torch.nn.Linear(hidden_features, 3 * heads * head_dim, bias=False).to("cuda", dtype)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=16,
        kv_chunk_tokens=16,
        block_m=16,
        block_n=16,
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    runner = ProjectedAttentionRunner(
        plan,
        ProjectionPipelineConfig(projection_tile_tokens=13),
    )

    def fail_projection(tile, start, stop):
        del tile, start, stop
        raise RuntimeError("injected projection failure")

    with pytest.raises(RuntimeError, match="injected projection failure"):
        runner.project_qkv_to_host(hidden, fail_projection)

    def project_qkv(tile, start, stop):
        del start, stop
        projected = linear(tile).view(-1, 3, heads, head_dim)
        return tuple(projected[:, index].contiguous() for index in range(3))

    q, k, v = runner.project_qkv_to_host(hidden, project_qkv)
    expected = linear(hidden.to("cuda")).view(-1, 3, heads, head_dim).cpu()
    torch.testing.assert_close(q, expected[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(k, expected[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(v, expected[:, 2], atol=0, rtol=0)


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
    plan = build_attention_plan(
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
        ProjectionPipelineConfig(projection_tile_tokens=96),
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


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
def test_projection_workspace_resets_without_device_wide_synchronization(monkeypatch):
    dtype = torch.float16
    tokens = 19
    hidden_features = 24
    heads = 2
    head_dim = 16
    inner = heads * head_dim
    hidden_cpu = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=8,
        kv_chunk_tokens=8,
        block_m=16,
        block_n=16,
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    runner = ProjectedAttentionRunner(
        plan,
        ProjectionPipelineConfig(
            projection_tile_tokens=7,
            num_projection_buffers=2,
        ),
    )
    qkv_linear = torch.nn.Linear(hidden_features, 3 * inner, bias=False).to(
        device="cuda:0",
        dtype=dtype,
    )
    callback_streams = []

    def project_qkv(hidden, start, stop):
        del start, stop
        callback_streams.append(torch.cuda.current_stream("cuda:0"))
        qkv = qkv_linear(hidden).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def reject_device_sync(*_args, **_kwargs):
        raise AssertionError("projection must not synchronize the complete CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", reject_device_sync)
    runner.project_qkv_to_host(hidden_cpu, project_qkv)

    workspace = runner._projection_workspace
    assert workspace is not None
    assert callback_streams
    assert all(stream == workspace.compute_stream for stream in callback_streams)
    assert workspace.busy == [False, False]
    assert workspace.keepalive == [None, None]

    callback_calls = 0

    def fail_second_tile(hidden, start, stop):
        nonlocal callback_calls
        callback_calls += 1
        result = project_qkv(hidden, start, stop)
        if callback_calls == 2:
            raise RuntimeError("injected projection failure")
        return result

    with pytest.raises(RuntimeError, match="injected projection failure"):
        runner.project_qkv_to_host(hidden_cpu, fail_second_tile)

    assert workspace.busy == [False, False]
    assert workspace.keepalive == [None, None]
    runner.project_qkv_to_host(hidden_cpu, project_qkv)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
def test_output_transform_rejects_a_tensor_on_the_wrong_cuda_device():
    dtype = torch.float16
    tokens = 17
    heads = 2
    head_dim = 16
    q = torch.randn(tokens, heads, head_dim, dtype=dtype, pin_memory=True)
    k = torch.randn_like(q, pin_memory=True)
    v = torch.randn_like(q, pin_memory=True)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=8,
        kv_chunk_tokens=8,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    runner = ProjectedAttentionRunner(plan).attention
    out = torch.empty((tokens, heads * head_dim), dtype=dtype, pin_memory=True)

    def wrong_device_output(attention, start, stop):
        del attention
        return torch.empty((stop - start, heads * head_dim), dtype=dtype, device="cuda:1")

    with pytest.raises(ValueError, match="planned CUDA device"):
        runner.run_with_device_output(
            q,
            k,
            v,
            cu,
            cu,
            output_transform=wrong_device_output,
            out=out,
        )

    actual = runner.run_with_device_output(
        q,
        k,
        v,
        cu,
        cu,
        output_transform=lambda attention, start, stop: attention,
        out=out,
    )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu,
        cu,
        q_chunk_tokens=8,
        kv_chunk_tokens=8,
        device="cuda:0",
    ).reshape(tokens, -1)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
def test_output_consumer_failure_synchronizes_and_allows_runner_reuse():
    dtype = torch.float16
    tokens = 17
    heads = 2
    head_dim = 16
    q = torch.randn(tokens, heads, head_dim, dtype=dtype, pin_memory=True)
    k = torch.randn_like(q, pin_memory=True)
    v = torch.randn_like(q, pin_memory=True)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=8,
        kv_chunk_tokens=8,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    runner = ProjectedAttentionRunner(plan).attention

    class FailingConsumer:
        def __init__(self):
            self.stream = torch.cuda.Stream(device="cuda:0")
            self.ready = torch.cuda.Event()
            self.output = torch.empty((tokens, heads * head_dim), dtype=dtype, device="cuda:0")
            self.synchronized = False

        def __call__(self, attention, start, stop):
            self.ready.record(torch.cuda.current_stream("cuda:0"))
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.ready)
                self.output[start:stop].copy_(attention)
            raise RuntimeError("injected consumer failure")

        def finish(self):
            raise AssertionError("finish must not run after a tile failure")

        def synchronize(self):
            self.stream.synchronize()
            self.synchronized = True

    consumer = FailingConsumer()
    with pytest.raises(RuntimeError, match="injected consumer failure"):
        runner.run_with_device_consumer(q, k, v, cu, cu, output_consumer=consumer)
    assert consumer.synchronized

    out = torch.empty((tokens, heads * head_dim), dtype=dtype, pin_memory=True)
    actual = runner.run_with_device_output(
        q,
        k,
        v,
        cu,
        cu,
        output_transform=lambda attention, start, stop: attention,
        out=out,
    )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu,
        cu,
        q_chunk_tokens=8,
        kv_chunk_tokens=8,
        device="cuda:0",
    ).reshape(tokens, -1)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
