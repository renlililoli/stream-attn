"""NVFP4 approximation tests; exact partition mechanics are tested separately."""

import math

import pytest
import torch

from seqattn_core import StreamingAttentionConfig, StreamingAttentionRunner, build_attention_plan
from seqattn_core.kernels import initialize_split_attention_state, merge_split_attention_state
from seqattn_core.streaming.sage3_backend import Sage3State, sage3_is_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not sage3_is_available(),
    reason="requires SM120 and the patched SageAttention3 CUDA image",
)


@pytest.mark.parametrize("tokens,keys,dim", [(128, 128, 64), (129, 131, 128), (1, 1, 64)])
@torch.inference_mode()
def test_sage3_lse_has_original_score_origin_and_does_not_mutate_inputs(tokens, keys, dim):
    torch.manual_seed(31)
    q = torch.randn(tokens, 2, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.full((keys, 2, dim), 3.0, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    copies = [x.clone() for x in (q, k, v)]
    state = Sage3State()
    state.prepare_query(q)
    output, lse = state.partial(k, v, dim**-0.5)
    expected = math.log(keys) + 3 * q.float().sum(-1).T * dim**-0.5
    torch.testing.assert_close(lse[0], expected, rtol=1e-5, atol=2e-4)
    assert torch.isfinite(output).all()
    assert all(torch.equal(a, b) for a, b in zip(copies, (q, k, v)))


@torch.inference_mode()
def test_sage3_partition_merge_matches_fp32_lse_weighted_oracle():
    torch.manual_seed(41)
    q = torch.randn(129, 2, 64, device="cuda", dtype=torch.bfloat16)
    state = Sage3State()
    state.prepare_query(q)
    parts = []
    for keys, shift in [(128, -2), (131, 2), (1, 0)]:
        k = torch.randn(keys, 2, 64, device="cuda", dtype=q.dtype) + shift
        parts.append(state.partial(k, torch.randn_like(k), 64**-0.5))
    accumulator = torch.empty(1, 129, 2, 64, device="cuda", dtype=torch.float32)
    lse_state = torch.empty(1, 129, 2, device="cuda", dtype=torch.float32)
    for i, (out, lse) in enumerate(parts):
        combine = initialize_split_attention_state if i == 0 else merge_split_attention_state
        combine(out, lse, accumulator, lse_state)
    lses = torch.stack([part[1] for part in parts])
    weights = lses.softmax(0).transpose(-1, -2).unsqueeze(-1)
    expected = (weights * torch.stack([part[0].float() for part in parts])).sum(0)
    torch.testing.assert_close(accumulator, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(lse_state, lses.logsumexp(0).transpose(-1, -2), rtol=1e-6, atol=2e-6)


@torch.inference_mode()
def test_sage3_streaming_preserves_segments_and_tail_masks(monkeypatch):
    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    plan = build_attention_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cuda",
        max_q_tokens=258,
        max_kv_tokens=258,
        config=StreamingAttentionConfig(q_chunk_tokens=128, kv_chunk_tokens=128),
    )
    runner = StreamingAttentionRunner(plan)
    assert runner.backend == "sage3"
    q = torch.ones(258, 2, 64, dtype=torch.bfloat16).pin_memory()
    k = q.clone().pin_memory()
    v = q.clone().pin_memory()
    # Within each packed segment, the last single-token partition dominates.
    k[:128] = -2
    k[128] = 2
    k[129:257] = -2
    k[257] = 2
    v.zero_()
    v[128] = 1
    v[257] = 3
    bounds = torch.tensor([0, 129, 258], dtype=torch.int32)
    result = runner(q, k, v, bounds, bounds)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        result[:129].float(), torch.ones_like(result[:129]).float(), rtol=0.08, atol=0.08
    )
    torch.testing.assert_close(
        result[129:].float(), torch.full_like(result[129:], 3).float(), rtol=0.08, atol=0.08
    )
    assert plan.backend_workspace_bytes > 0


@torch.inference_mode()
def test_sage3_zero_inputs_have_finite_uniform_lse():
    x = torch.zeros(128, 2, 64, device="cuda", dtype=torch.bfloat16)
    state = Sage3State()
    state.prepare_query(x)
    out, lse = state.partial(x, x, 64**-0.5)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(128)), rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_explicit_backend_overrides_container_default_and_budget_is_reserved(monkeypatch):
    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    kwargs = {
        "q_heads": 2,
        "kv_heads": 2,
        "head_dim": 64,
        "dtype": torch.bfloat16,
        "device": "cuda",
        "max_q_tokens": 128,
        "max_kv_tokens": 128,
    }
    regular = build_attention_plan(**kwargs, config=StreamingAttentionConfig(backend="triton"))
    packed = build_attention_plan(**kwargs, config=StreamingAttentionConfig(backend="sage3"))
    assert StreamingAttentionRunner(regular).backend == "triton"
    assert (
        packed.estimated_workspace_bytes
        == regular.estimated_workspace_bytes + packed.backend_workspace_bytes
    )
    assert (
        StreamingAttentionRunner(packed).plan.estimated_workspace_bytes
        == packed.estimated_workspace_bytes
    )
    with pytest.raises(ValueError, match="workspace"):
        build_attention_plan(
            **kwargs,
            config=StreamingAttentionConfig(
                backend="sage3",
                q_chunk_tokens=128,
                workspace_budget_bytes=regular.estimated_workspace_bytes,
            ),
        )


@torch.inference_mode()
def test_partial_output_matches_sage3_single_partition_api():
    from sageattn3 import sageattn3_blackwell

    torch.manual_seed(17)
    q = torch.randn(129, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(131, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    state = Sage3State()
    state.prepare_query(q)
    out, _ = state.partial(k, v, 128**-0.5)
    expected = sageattn3_blackwell(
        q.unsqueeze(0).transpose(1, 2).contiguous(),
        k.unsqueeze(0).transpose(1, 2).contiguous().clone(),
        v.unsqueeze(0).transpose(1, 2).contiguous(),
    )
    torch.testing.assert_close(out, expected.transpose(1, 2).contiguous(), rtol=0, atol=0)


@torch.inference_mode()
def test_causal_auto_falls_back_and_explicit_sage_rejects(monkeypatch):
    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    kwargs = {
        "q_heads": 2,
        "kv_heads": 2,
        "head_dim": 64,
        "dtype": torch.bfloat16,
        "device": "cuda",
        "max_q_tokens": 128,
        "max_kv_tokens": 128,
    }
    q = torch.randn(128, 2, 64, dtype=torch.bfloat16).pin_memory()
    k, v = q.clone().pin_memory(), q.clone().pin_memory()
    bounds = torch.tensor([0, 128], dtype=torch.int32)
    auto = StreamingAttentionRunner(
        build_attention_plan(**kwargs, config=StreamingAttentionConfig())
    )
    triton = StreamingAttentionRunner(
        build_attention_plan(**kwargs, config=StreamingAttentionConfig(backend="triton"))
    )
    actual = auto(q, k, v, bounds, bounds, causal=True)
    expected = triton(q, k, v, bounds, bounds, causal=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    explicit = StreamingAttentionRunner(
        build_attention_plan(**kwargs, config=StreamingAttentionConfig(backend="sage3"))
    )
    with pytest.raises(ValueError, match="causal"):
        explicit(q, k, v, bounds, bounds, causal=True)


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
@torch.inference_mode()
def test_sage3_complete_h3_consumer_and_reuse(mode, monkeypatch):
    from seqattn_core.dit.minimax_h3 import (
        H3BlockOps,
        H3Config,
        H3MaterializedProjection,
        H3RecomputeProjection,
        H3SequenceMeta,
        build_h3_runner,
    )

    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    plan = build_attention_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cuda",
        max_q_tokens=258,
        max_kv_tokens=258,
        config=StreamingAttentionConfig(
            q_chunk_tokens=128, kv_chunk_tokens=128, output_mode="device_consumer"
        ),
    )
    runner = build_h3_runner(
        plan,
        hidden_features=128,
        config=H3Config(execution_mode=mode, projection_tile_tokens=128, ffn_tile_tokens=80),
    )
    host = torch.randn(258, 128, dtype=torch.bfloat16).pin_memory()
    dest = torch.empty_like(host, pin_memory=True)
    ranges = []

    def epilogue(out, residual, start, stop):
        return residual[start:stop].to("cuda", non_blocking=True).add_(out)

    def ffn(post, start, stop):
        ranges.append((start, stop))
        return post

    ops = H3BlockOps(epilogue, ffn)
    meta = H3SequenceMeta(torch.tensor([0, 129, 258], dtype=torch.int32))
    if mode == "materialized":

        def project(tile, start, stop):
            q = tile.view(-1, 2, 64)
            return q, q, q

        projection = H3MaterializedProjection(project)
        run = lambda: runner.run_block_(host, meta, projection, ops)
        assert runner.projected_attention.attention.backend == "sage3"
    else:

        def project_q(tile, out, start, stop):
            out.copy_(tile.view(-1, 2, 64))

        def project_kv(tile, k, v, start, stop):
            k.copy_(tile.view(-1, 2, 64))
            v.copy_(tile.view(-1, 2, 64))

        projection = H3RecomputeProjection(project_q, project_kv)
        run = lambda: runner.run_block(host, dest, meta, projection, ops)
        assert runner.recomputed_attention.attention.backend == "sage3"
    for _ in range(2):
        ranges.clear()
        result = run()
        assert torch.isfinite(result).all()
        assert ranges == [(0, 80), (80, 160), (160, 240), (240, 258)]


@torch.inference_mode()
def test_sol_configuration_uses_sage3_for_dense_policy_and_triton_for_sparse(monkeypatch):
    from seqattn_core.dit.minimax_h3 import (
        H3BlockOps,
        H3Config,
        H3DenoisingStep,
        H3MaterializedProjection,
        H3SequenceMeta,
        build_h3_runner,
    )

    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    tokens = 258
    plan = build_attention_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=StreamingAttentionConfig(
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            output_mode="device_consumer",
        ),
    )
    runner = build_h3_runner(
        plan,
        hidden_features=256,
        config=H3Config(
            attention_mode="sol_streaming",
            projection_tile_tokens=128,
            ffn_tile_tokens=128,
            sol_first_dense_step_fraction=0.5,
            sol_first_dense_layers=1,
        ),
    )
    assert runner.projected_attention.attention.backend == "sage3"
    assert runner.sol_attention.dense_runner.backend == "triton"
    assert runner.plan.estimated_workspace_bytes > (
        runner.projected_attention.plan.estimated_workspace_bytes
        + runner.sol_attention.plan.estimated_workspace_bytes
    )

    source = torch.randn(tokens, 256, dtype=torch.bfloat16).pin_memory()
    baseline = source.clone()
    meta = H3SequenceMeta(
        torch.tensor([0, 129, tokens], dtype=torch.int32),
        exact_prefix_tokens=(64, 64),
    )

    def project(tile, start, stop):
        q = tile.view(stop - start, 2, 128)
        return q, q, q

    ops = H3BlockOps(
        lambda output, residual, start, stop: (
            residual[start:stop].to("cuda", non_blocking=True).add_(output)
        ),
        lambda post, start, stop: post,
    )
    projection = H3MaterializedProjection(project)

    # Early step follows the unchanged dense policy, now through Sage3.
    runner.run_block_(
        source,
        meta,
        projection,
        ops,
        block_index=2,
        denoising_step=H3DenoisingStep(0, 4),
    )
    assert torch.isfinite(source).all()

    # Late step follows the existing Sol route/summary implementation on Triton.
    source.copy_(baseline)
    stats = runner.run_block_(
        source,
        meta,
        projection,
        ops,
        block_index=2,
        denoising_step=H3DenoisingStep(3, 4),
    )
    assert torch.isfinite(stats).all()
