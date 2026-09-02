import math
from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    RecomputedAttentionRunner,
    RecomputedAttentionStats,
    StreamingAttentionConfig,
    build_attention_plan,
)
from seqattn_core.dit.minimax_h3 import (
    H3BlockOps,
    H3Config,
    H3DiTStats,
    H3MaterializedProjection,
    H3MaterializedRunner,
    H3RecomputeProjection,
    H3RecomputeRunner,
    H3SequenceMeta,
    estimate_h3_materialized_aux_workspace_bytes,
    estimate_h3_recompute_aux_workspace_bytes,
    load_h3_config,
)
from seqattn_core.kernels import triton_is_available


def test_h3_workspace_estimates_and_sequence_validation():
    materialized = (2 * 31 * 48 + 3 * 43 * 48) * 2
    assert (
        estimate_h3_materialized_aux_workspace_bytes(
            hidden_features=48,
            dtype=torch.bfloat16,
            projection_tile_tokens=31,
            num_projection_buffers=2,
            ffn_tile_tokens=43,
        )
        == materialized
    )

    recompute = (41 * 48 + 3 * 43 * 48) * 2
    assert (
        estimate_h3_recompute_aux_workspace_bytes(
            hidden_features=48,
            dtype=torch.bfloat16,
            hidden_staging_tokens=41,
            ffn_tile_tokens=43,
        )
        == recompute
    )

    meta = H3SequenceMeta(torch.tensor([0, 3, 3, 7], dtype=torch.int32))
    meta.validate(7)
    with pytest.raises(ValueError, match="span"):
        meta.validate(8)
    with pytest.raises(ValueError, match="int32"):
        H3SequenceMeta(torch.tensor([0, 7], dtype=torch.int64)).validate(7)
    with pytest.raises(ValueError, match="non-decreasing"):
        H3SequenceMeta(torch.tensor([0, 5, 4, 7], dtype=torch.int32)).validate(7)


def test_h3_config_from_toml(tmp_path):
    path = tmp_path / "seqattn.toml"
    path.write_text(
        "[attention]\nbackend = 'auto'\n\n"
        "[minimax_h3]\nprojection_tile_tokens = 1024\nffn_tile_tokens = 512\n"
    )
    assert load_h3_config(path) == H3Config(
        projection_tile_tokens=1024,
        ffn_tile_tokens=512,
    )

    path.write_text("[minimax_h3]\nprojection_tile_tokens = 0\n")
    with pytest.raises(ValueError, match="projection_tile_tokens"):
        load_h3_config(path)


def test_h3_config_defaults_to_tuned_block_tiles(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQATTN_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert load_h3_config() == H3Config(
        projection_tile_tokens=4096,
        ffn_tile_tokens=4096,
    )


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


def _attention_tile_ranges(bounds, *, q_chunk_tokens, kv_chunk_tokens):
    q_ranges = []
    kv_ranges = []
    for segment_start, segment_stop in pairwise(bounds):
        for q_start in range(segment_start, segment_stop, q_chunk_tokens):
            q_ranges.append((q_start, min(q_start + q_chunk_tokens, segment_stop)))
            kv_ranges.extend(
                (kv_start, min(kv_start + kv_chunk_tokens, segment_stop))
                for kv_start in range(segment_start, segment_stop, kv_chunk_tokens)
            )
    return q_ranges, kv_ranges


def _build_block_modules(*, hidden_features, heads, head_dim, mlp_features, dtype, device):
    inner = heads * head_dim
    qkv_linear = torch.nn.Linear(hidden_features, inner * 3, bias=False).to(
        device=device, dtype=dtype
    )
    out_linear = torch.nn.Linear(inner, hidden_features, bias=False).to(device=device, dtype=dtype)
    fc1 = torch.nn.Linear(hidden_features, mlp_features * 2, bias=False).to(
        device=device, dtype=dtype
    )
    fc2 = torch.nn.Linear(mlp_features, hidden_features, bias=False).to(device=device, dtype=dtype)
    return qkv_linear, out_linear, fc1, fc2


def _full_block(hidden, qkv_linear, out_linear, fc1, fc2, *, heads, head_dim, bounds):
    tokens = hidden.shape[0]
    inner = heads * head_dim
    qkv = qkv_linear(hidden).view(tokens, 3, heads, head_dim)
    q, k, v = (qkv[:, index] for index in range(3))
    attention = _segmented_attention(q, k, v, bounds, head_dim**-0.5)
    post_attention = out_linear(attention.reshape(tokens, inner)).add(hidden)
    gate, up = fc1(post_attention).chunk(2, dim=-1)
    return post_attention + fc2(torch.nn.functional.silu(gate) * up)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_h3_materialized_runner_matches_full_gpu():
    torch.manual_seed(211)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens = 97
    hidden_features = 48
    heads = 4
    head_dim = 16
    mlp_features = 80
    bounds = [0, 37, 37, tokens]
    cu = torch.tensor(bounds, dtype=torch.int32)
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = hidden.clone()
    qkv_linear, out_linear, fc1, fc2 = _build_block_modules(
        hidden_features=hidden_features,
        heads=heads,
        head_dim=head_dim,
        mlp_features=mlp_features,
        dtype=dtype,
        device=device,
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
        projection_tile_tokens=31,
        num_projection_buffers=2,
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    runner = H3MaterializedRunner(
        ProjectedAttentionRunner(plan, projection_config),
        hidden_features=hidden_features,
        ffn_tile_tokens=43,
    )

    def project_qkv(tile, start, stop):
        del start, stop
        qkv = qkv_linear(tile).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def attention_epilogue(attention, residual_host, start, stop):
        residual = residual_host[start:stop].to(device, non_blocking=True)
        return out_linear(attention).add_(residual)

    def mlp(post_attention, start, stop):
        del start, stop
        gate, up = fc1(post_attention).chunk(2, dim=-1)
        return post_attention.add_(fc2(torch.nn.functional.silu(gate).mul_(up)))

    stats = H3DiTStats()
    pointer = hidden.data_ptr()
    result = runner.run_block_(
        hidden,
        H3SequenceMeta(cu),
        H3MaterializedProjection(project_qkv),
        H3BlockOps(attention_epilogue, mlp),
        softmax_scale=head_dim**-0.5,
        stats=stats,
    )
    expected = _full_block(
        original.to(device),
        qkv_linear,
        out_linear,
        fc1,
        fc2,
        heads=heads,
        head_dim=head_dim,
        bounds=bounds,
    )

    assert result.data_ptr() == pointer
    torch.testing.assert_close(hidden, expected.cpu(), atol=7e-2, rtol=1e-2)
    assert runner.plan.projection_tile_tokens == 31
    assert stats.qkv_storage_policy == "materialized"
    assert stats.blocks == 1
    assert stats.ffn_tiles == math.ceil(tokens / 43)
    assert stats.ffn_cross_q_boundaries >= 1
    assert stats.projection.attention.d2h_bytes == 0
    assert stats.qkv_host_bytes_peak == 3 * tokens * heads * head_dim * hidden.element_size()


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_h3_recompute_large_tiles_match_full_gpu_and_ping_pong():
    torch.manual_seed(307)
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
    source = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    source_before = source.clone()
    destination = torch.empty_like(source, pin_memory=True)
    qkv_linear, out_linear, fc1, fc2 = _build_block_modules(
        hidden_features=hidden_features,
        heads=heads,
        head_dim=head_dim,
        mlp_features=mlp_features,
        dtype=dtype,
        device=device,
    )

    attention_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=29,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    recomputed = RecomputedAttentionRunner(
        plan,
        hidden_features=hidden_features,
    )
    runner = H3RecomputeRunner(recomputed, ffn_tile_tokens=19)
    q_ranges = []
    kv_ranges = []

    def project_q(tile, destination_q, start, stop):
        q_ranges.append((start, stop))
        q = qkv_linear(tile)[:, :inner].view(-1, heads, head_dim)
        destination_q.copy_(q)

    def project_kv(tile, destination_k, destination_v, start, stop):
        kv_ranges.append((start, stop))
        projected = qkv_linear(tile)
        k = projected[:, inner : 2 * inner].view(-1, heads, head_dim)
        v = projected[:, 2 * inner :].view(-1, heads, head_dim)
        destination_k.copy_(k)
        destination_v.copy_(v)

    def attention_epilogue(attention, residual_host, start, stop):
        residual = residual_host[start:stop].to(device, non_blocking=True)
        return out_linear(attention).add_(residual)

    def mlp(post_attention, start, stop):
        del start, stop
        gate, up = fc1(post_attention).chunk(2, dim=-1)
        return post_attention.add_(fc2(torch.nn.functional.silu(gate).mul_(up)))

    projection = H3RecomputeProjection(project_q, project_kv)
    ops = H3BlockOps(attention_epilogue, mlp)
    with pytest.raises(ValueError, match="distinct source and destination"):
        runner.run_block(source, source, H3SequenceMeta(cu), projection, ops)

    stats = H3DiTStats()
    result = runner.run_block(
        source,
        destination,
        H3SequenceMeta(cu),
        projection,
        ops,
        softmax_scale=head_dim**-0.5,
        stats=stats,
    )
    expected = _full_block(
        source_before.to(device),
        qkv_linear,
        out_linear,
        fc1,
        fc2,
        heads=heads,
        head_dim=head_dim,
        bounds=bounds,
    )

    assert result.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(source, source_before)
    torch.testing.assert_close(destination, expected.cpu(), atol=7e-2, rtol=1e-2)
    assert not hasattr(runner.plan, "projection_tile_tokens")
    assert runner.plan.hidden_staging_tokens == max(
        runner.plan.q_chunk_tokens,
        runner.plan.kv_chunk_tokens,
    )
    assert recomputed.workspace.hidden.shape == (
        runner.plan.hidden_staging_tokens,
        hidden_features,
    )
    assert not any(hasattr(recomputed, name) for name in ("q_cpu", "k_cpu", "v_cpu"))
    expected_q_ranges, expected_kv_ranges = _attention_tile_ranges(
        bounds,
        q_chunk_tokens=runner.plan.q_chunk_tokens,
        kv_chunk_tokens=runner.plan.kv_chunk_tokens,
    )
    assert runner.plan.q_chunk_tokens != runner.plan.kv_chunk_tokens
    assert q_ranges == expected_q_ranges
    assert kv_ranges == expected_kv_ranges
    assert stats.recompute.q_projection_chunks == len(q_ranges)
    assert stats.recompute.kv_projection_chunks == len(kv_ranges)
    assert stats.recompute.attention.q_chunks == len(q_ranges)
    assert stats.recompute.attention.kv_tiles == len(kv_ranges)
    assert stats.recompute.attention.h2d_bytes == 0
    assert stats.recompute.qkv_host_bytes == 0
    assert stats.qkv_host_bytes_peak == 0
    assert stats.hidden_host_bytes_peak == 2 * source.numel() * source.element_size()

    stack_source = source_before.clone().pin_memory()
    stack_scratch = torch.empty_like(stack_source, pin_memory=True)
    stack_result = runner.run_blocks_(
        stack_source,
        stack_scratch,
        H3SequenceMeta(cu),
        [(projection, ops), (projection, ops)],
        softmax_scale=head_dim**-0.5,
    )
    expected_stack = _full_block(
        expected,
        qkv_linear,
        out_linear,
        fc1,
        fc2,
        heads=heads,
        head_dim=head_dim,
        bounds=bounds,
    )
    assert stack_result.data_ptr() == stack_source.data_ptr()
    torch.testing.assert_close(stack_result, expected_stack.cpu(), atol=1e-1, rtol=2e-2)


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_recomputed_attention_projection_failure_allows_runner_reuse():
    torch.manual_seed(397)
    device = torch.device("cuda")
    dtype = torch.float16
    tokens = 17
    hidden_features = 24
    heads = 2
    head_dim = 16
    inner = heads * head_dim
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    qkv = torch.nn.Linear(hidden_features, 3 * inner, bias=False).to(device, dtype)
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
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    runner = RecomputedAttentionRunner(
        plan,
        hidden_features=hidden_features,
    )

    def project_q(tile, destination, start, stop):
        del start, stop
        destination.copy_(qkv(tile)[:, :inner].view(-1, heads, head_dim))

    def project_kv(tile, destination_k, destination_v, start, stop):
        del start, stop
        projected = qkv(tile)
        destination_k.copy_(projected[:, inner : 2 * inner].view(-1, heads, head_dim))
        destination_v.copy_(projected[:, 2 * inner :].view(-1, heads, head_dim))

    kv_calls = 0

    def fail_second_kv(tile, destination_k, destination_v, start, stop):
        nonlocal kv_calls
        kv_calls += 1
        project_kv(tile, destination_k, destination_v, start, stop)
        if kv_calls == 2:
            raise RuntimeError("injected KV projection failure")

    class Collector:
        def __init__(self):
            self.output = torch.empty((tokens, inner), dtype=dtype, device=device)

        def __call__(self, attention, start, stop):
            self.output[start:stop].copy_(attention)

        def finish(self):
            return None

        def synchronize(self):
            torch.cuda.current_stream(device).synchronize()

    with pytest.raises(RuntimeError, match="injected KV projection failure"):
        runner.run_with_device_consumer(
            hidden,
            cu,
            project_q=project_q,
            project_kv=fail_second_kv,
            output_consumer=Collector(),
        )
    assert not runner.workspace.hidden_has_pending_compute

    collector = Collector()
    stats = RecomputedAttentionStats()
    runner.run_with_device_consumer(
        hidden,
        cu,
        project_q=project_q,
        project_kv=project_kv,
        output_consumer=collector,
        stats=stats,
    )
    projected = qkv(hidden.to(device))
    q = projected[:, :inner].view(tokens, heads, head_dim)
    k = projected[:, inner : 2 * inner].view(tokens, heads, head_dim)
    v = projected[:, 2 * inner :].view(tokens, heads, head_dim)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        scale=head_dim**-0.5,
    ).squeeze(0)
    expected = expected.transpose(0, 1).reshape(tokens, inner)
    torch.testing.assert_close(collector.output, expected, atol=3e-2, rtol=3e-2)
    assert stats.raw_attention_roundtrip_bytes_avoided == (
        2 * tokens * inner * hidden.element_size()
    )


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_h3_recompute_runner_reuse_has_bounded_allocator_growth():
    torch.manual_seed(401)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens = 73
    hidden_features = 48
    heads = 4
    head_dim = 16
    inner = heads * head_dim
    source = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    destination = torch.empty_like(source, pin_memory=True)
    qkv = torch.nn.Linear(hidden_features, 3 * inner, bias=False).to(device, dtype)
    out = torch.nn.Linear(inner, hidden_features, bias=False).to(device, dtype)
    attention_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=29,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=attention_config,
    )
    runner = H3RecomputeRunner(
        RecomputedAttentionRunner(
            plan,
            hidden_features=hidden_features,
        ),
        ffn_tile_tokens=19,
    )

    def project_q(tile, destination, start, stop):
        del start, stop
        destination.copy_(qkv(tile)[:, :inner].view(-1, heads, head_dim))

    def project_kv(tile, destination_k, destination_v, start, stop):
        del start, stop
        projected = qkv(tile)
        destination_k.copy_(projected[:, inner : 2 * inner].view(-1, heads, head_dim))
        destination_v.copy_(projected[:, 2 * inner :].view(-1, heads, head_dim))

    def attention_epilogue(attention, residual_host, start, stop):
        return out(attention).add_(residual_host[start:stop].to(device, non_blocking=True))

    projection = H3RecomputeProjection(project_q, project_kv)
    ops = H3BlockOps(attention_epilogue, lambda tile, start, stop: tile)
    meta = H3SequenceMeta(torch.tensor([0, tokens], dtype=torch.int32))

    def fail_project_q(tile, destination, start, stop):
        del tile, destination, start, stop
        raise RuntimeError("injected recompute projection failure")

    with pytest.raises(RuntimeError, match="injected recompute projection failure"):
        runner.run_block(
            source,
            destination,
            meta,
            H3RecomputeProjection(fail_project_q, project_kv),
            ops,
        )

    runner.run_block(source, destination, meta, projection, ops)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(10):
        runner.run_block(source, destination, meta, projection, ops)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline + 8 * 2**20
