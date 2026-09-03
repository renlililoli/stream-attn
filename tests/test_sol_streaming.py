import math
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    build_attention_plan,
)
from seqattn_core.sparse import (
    SolStreamingAttentionRunner,
    SolStreamingStats,
    build_sol_streaming_plan,
    sol_streaming_reference,
)


def _dense_segmented(q, k, v, bounds, scale):
    output = torch.empty_like(q)
    for start, stop in pairwise(bounds):
        if start == stop:
            continue
        scores = (
            torch.einsum(
                "thd,shd->hts",
                q[start:stop].float(),
                k[start:stop].float(),
            )
            * scale
        )
        probabilities = torch.softmax(scores, dim=-1)
        result = torch.einsum("hts,shd->thd", probabilities, v[start:stop].float())
        output[start:stop].copy_(result.to(output.dtype))
    return output


def _reference_inputs(tokens=193, heads=2):
    generator = torch.Generator().manual_seed(1701)
    shape = (tokens, heads, 128)
    return tuple(torch.randn(shape, generator=generator).to(torch.bfloat16) for _ in range(3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_sol_int8_encoder_preserves_tail_summaries_and_quantization_bounds():
    from seqattn_core.kernels.sol_preprocess import encode_sol_kv

    _, k_cpu, v_cpu = _reference_inputs(tokens=65, heads=2)
    k = k_cpu.cuda()
    v = v_cpu.cuda()
    encoded = encode_sol_kv(k, v)
    k_quantized, v_quantized, k_scales, v_scales, k_centroids, value_sums = encoded

    assert k_quantized.shape == k.shape
    assert v_quantized.shape == v.shape
    assert k_quantized.dtype == v_quantized.dtype == torch.int8
    assert k_scales.shape == v_scales.shape == (2, 2)
    assert k_scales.dtype == v_scales.dtype == torch.float16
    assert k_centroids.shape == value_sums.shape == (2, 2, 128)

    for source, quantized, scales in (
        (k, k_quantized, k_scales),
        (v, v_quantized, v_scales),
    ):
        token_scales = scales.repeat_interleave(64, dim=0)[: source.shape[0]]
        reconstructed = quantized.float() * token_scales[:, :, None].float()
        error = (reconstructed - source.float()).abs()
        # The quantizer computes its scale in FP32 and stores it as FP16. The
        # bound therefore includes both half-step rounding and scale storage error.
        assert torch.all(error <= token_scales[:, :, None].float() * 0.57 + 1.0e-6)

    torch.testing.assert_close(
        k_centroids[0].float(),
        k[:64].float().mean(dim=0),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        value_sums[0].float(),
        v[:64].float().sum(dim=0),
        atol=2e-1,
        rtol=2e-2,
    )
    torch.testing.assert_close(k_centroids[1], k[64], atol=0, rtol=0)
    torch.testing.assert_close(value_sums[1], v[64], atol=0, rtol=0)


def test_sol_reference_all_exact_matches_segmented_dense():
    q, k, v = _reference_inputs()
    bounds = [0, 65, 65, 193]
    cu = torch.tensor(bounds, dtype=torch.int32)
    scale = 128**-0.5
    got = sol_streaming_reference(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=(0, 0, 0),
        tau=-1000.0,
        softmax_scale=scale,
    )
    expected = _dense_segmented(q, k, v, bounds, scale)
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


def test_sol_reference_prefix_queries_and_packed_segments_are_exact():
    q, k, v = _reference_inputs(tokens=320, heads=1)
    bounds = [0, 192, 320]
    cu = torch.tensor(bounds, dtype=torch.int32)
    prefixes = (65, 0)
    scale = 128**-0.5
    got = sol_streaming_reference(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=prefixes,
        tau=1000.0,
        softmax_scale=scale,
    )
    dense = _dense_segmented(q, k, v, bounds, scale)
    # Prefixes round outward to complete 64-token query blocks.
    torch.testing.assert_close(got[:128], dense[:128], atol=2e-2, rtol=2e-2)

    first = sol_streaming_reference(
        q[:192],
        k[:192],
        v[:192],
        torch.tensor([0, 192], dtype=torch.int32),
        exact_prefix_tokens=(65,),
        tau=1000.0,
        softmax_scale=scale,
    )
    second = sol_streaming_reference(
        q[192:],
        k[192:],
        v[192:],
        torch.tensor([0, 128], dtype=torch.int32),
        exact_prefix_tokens=(0,),
        tau=1000.0,
        softmax_scale=scale,
    )
    torch.testing.assert_close(got, torch.cat((first, second)), atol=0, rtol=0)


def test_sol_stats_reports_effective_density():
    stats = SolStreamingStats(exact_route_blocks=3, approximate_route_blocks=1)
    assert stats.effective_density == 0.75
    assert stats.as_dict()["effective_density"] == 0.75


def test_sol_q_ranges_balance_route_aligned_tail_without_exceeding_plan_chunk():
    from seqattn_core.sparse.runner import _balanced_q_ranges

    ranges = _balanced_q_ranges(100_000, 8192)
    sizes = [stop - start for start, stop in ranges]
    assert len(ranges) == math.ceil(100_000 / 8192)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100_000
    assert all(left[1] == right[0] for left, right in pairwise(ranges))
    assert all(start % 64 == 0 for start, _ in ranges)
    assert max(sizes) <= 8192
    assert min(sizes) >= 7680


class _CollectDeviceOutput:
    def __init__(self, tokens, heads, device):
        self.output = torch.empty((tokens, heads * 128), dtype=torch.bfloat16, device=device)

    def __call__(self, tile, start, stop):
        self.output[start:stop].copy_(tile)

    def finish(self):
        return None

    def synchronize(self):
        torch.cuda.synchronize(self.output.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tau", [-1000.0, 1.0])
def test_sol_streaming_cuda_matches_semantic_reference(tau):
    device = torch.device("cuda")
    tokens = 321
    heads = 2
    q, k, v = (tensor.pin_memory() for tensor in _reference_inputs(tokens=tokens, heads=heads))
    cu = torch.tensor([0, 193, tokens], dtype=torch.int32)
    prefixes = (65, 0)
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=128,
        dtype=torch.bfloat16,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            output_mode="device_consumer",
        ),
    )
    sol_plan = build_sol_streaming_plan(plan)
    dense = StreamingAttentionRunner(sol_plan.attention)
    runner = SolStreamingAttentionRunner(sol_plan, dense)
    runtime = dense._borrow_cuda_runtime()
    assert runner.workspace.dense is runtime.workspace
    assert runner._single_flight_lock is runtime.single_flight_lock
    consumer = _CollectDeviceOutput(tokens, heads, device)
    stats = SolStreamingStats()
    runner.run_with_device_consumer(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=prefixes,
        output_consumer=consumer,
        tau=tau,
        stats=stats,
    )
    expected = sol_streaming_reference(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=prefixes,
        tau=tau,
    )
    got = consumer.output.view(tokens, heads, 128).cpu()
    torch.testing.assert_close(got, expected, atol=5e-2, rtol=5e-2)
    assert stats.summary_kv_tokens == tokens
    assert stats.q_chunks == math.ceil(193 / 128) + 1
    assert stats.exact_route_blocks + stats.approximate_route_blocks > 0
    assert stats.kv_storage_dtype == "bf16"
    assert stats.backend == "sol_streaming:triton:bf16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sol_streaming_recovers_after_output_consumer_failure():
    device = torch.device("cuda")
    tokens = 193
    heads = 1
    q, k, v = (tensor.pin_memory() for tensor in _reference_inputs(tokens=tokens, heads=heads))
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=128,
        dtype=torch.bfloat16,
        device=device,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=128,
            kv_chunk_tokens=128,
            output_mode="device_consumer",
        ),
    )
    sol_plan = build_sol_streaming_plan(plan)
    runner = SolStreamingAttentionRunner(
        sol_plan,
        StreamingAttentionRunner(sol_plan.attention),
    )

    class FailingConsumer:
        def __init__(self):
            self.synchronized = False

        def __call__(self, tile, start, stop):
            del tile, start, stop
            raise RuntimeError("injected Sol consumer failure")

        def finish(self):
            raise AssertionError("finish must not run after a tile failure")

        def synchronize(self):
            torch.cuda.synchronize(device)
            self.synchronized = True

    failing = FailingConsumer()
    with pytest.raises(RuntimeError, match="injected Sol consumer failure"):
        runner.run_with_device_consumer(
            q,
            k,
            v,
            cu,
            exact_prefix_tokens=(0,),
            output_consumer=failing,
            tau=-1000.0,
        )
    assert failing.synchronized

    consumer = _CollectDeviceOutput(tokens, heads, device)
    runner.run_with_device_consumer(
        q,
        k,
        v,
        cu,
        exact_prefix_tokens=(0,),
        output_consumer=consumer,
        tau=-1000.0,
    )
    expected = _dense_segmented(q, k, v, [0, tokens], 128**-0.5)
    torch.testing.assert_close(
        consumer.output.view(tokens, heads, 128).cpu(),
        expected,
        atol=5e-2,
        rtol=5e-2,
    )


def _sol_compatible_plan(**changes):
    plan = build_attention_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        max_q_tokens=321,
        max_kv_tokens=321,
        config=StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=190,
            kv_chunk_tokens=150,
            block_m=16,
            block_n=16,
            num_warps=4,
            num_stages=1,
            output_mode="device_consumer",
        ),
    )
    return replace(plan, **changes)


def test_sol_plan_aligns_route_chunks_and_accounts_sparse_workspace():
    plan = build_sol_streaming_plan(_sol_compatible_plan())
    assert plan.route_block_tokens == 64
    assert plan.attention.q_chunk_tokens == 128
    assert plan.attention.kv_chunk_tokens == 128
    assert plan.max_q_blocks == 2
    assert plan.max_kv_blocks == 6
    assert plan.estimated_workspace_bytes == (
        plan.dense_workspace_bytes + plan.sparse_workspace_bytes
    )
    assert plan.attention.estimated_workspace_bytes == plan.estimated_workspace_bytes


def test_sol_plan_reduces_q_chunk_to_respect_workspace_budget():
    base = _sol_compatible_plan(q_chunk_tokens=256, kv_chunk_tokens=128)
    unconstrained = build_sol_streaming_plan(base)
    budget = unconstrained.estimated_workspace_bytes - 1
    constrained = build_sol_streaming_plan(replace(base, workspace_budget_bytes=budget))
    assert constrained.attention.q_chunk_tokens == 192
    assert constrained.estimated_workspace_bytes <= budget


def test_sol_plan_reduces_kv_chunk_after_q_reaches_minimum():
    base = _sol_compatible_plan(q_chunk_tokens=64, kv_chunk_tokens=256)
    unconstrained = build_sol_streaming_plan(base)
    budget = unconstrained.estimated_workspace_bytes - 1
    constrained = build_sol_streaming_plan(replace(base, workspace_budget_bytes=budget))
    assert constrained.attention.q_chunk_tokens == 64
    assert constrained.attention.kv_chunk_tokens == 192
    assert constrained.estimated_workspace_bytes <= budget


def test_sol_plan_rejects_insufficient_workspace_budget():
    with pytest.raises(ValueError, match="workspace budget"):
        build_sol_streaming_plan(
            _sol_compatible_plan(
                q_chunk_tokens=64,
                workspace_budget_bytes=1,
            )
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"device": torch.device("cpu")}, "CUDA device"),
        ({"output_mode": "host"}, "device_consumer"),
        ({"dtype": torch.float16}, "bfloat16"),
        ({"head_dim": 64}, "head_dim=128"),
        ({"kv_heads": 1}, "equal Q and K/V"),
        ({"backend": "fa2"}, "Triton backend"),
        ({"q_chunk_tokens": 32}, "64-token Sol route block"),
    ],
)
def test_sol_plan_rejects_unsupported_v1_contracts(changes, message):
    with pytest.raises(ValueError, match=message):
        build_sol_streaming_plan(_sol_compatible_plan(**changes))
