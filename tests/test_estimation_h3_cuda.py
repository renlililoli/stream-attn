"""Small real-CUDA H3 runner validation of the offline schedule and owned tensors."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_estimation_h3 import profile

from seqattn_core import StreamingAttentionConfig, build_attention_plan
from seqattn_core.dit.minimax_h3 import (
    H3BlockOps,
    H3Config,
    H3MaterializedProjection,
    H3RecomputeProjection,
    H3SequenceMeta,
    build_h3_runner,
)
from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3ExecutionConfig,
    build_h3_block_execution,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
@torch.inference_mode()
def test_real_h3_runner_matches_estimated_ranges_and_persistent_allocations(mode):
    torch.manual_seed(7)
    device, dtype = torch.device("cuda", torch.cuda.current_device()), torch.bfloat16
    shape = H3BlockShape((96, 65), 32, 64, 4, 64, rope_dim=32)
    config = H3ExecutionConfig(64, 64, 48, 80, execution_mode=mode)
    plan = build_attention_plan(
        q_heads=4,
        kv_heads=4,
        head_dim=64,
        dtype=dtype,
        device=device,
        max_q_tokens=161,
        max_kv_tokens=161,
        config=StreamingAttentionConfig(
            backend="triton", q_chunk_tokens=64, kv_chunk_tokens=64, output_mode="device_consumer"
        ),
    )
    runner = build_h3_runner(
        plan,
        hidden_features=32,
        config=H3Config(execution_mode=mode, projection_tile_tokens=48, ffn_tile_tokens=80),
    )
    weights = [
        torch.randn(n, k, device=device, dtype=dtype) * 0.02
        for n, k in ((768, 32), (32, 256), (128, 32), (32, 64))
    ]
    host = torch.randn(161, 32, dtype=dtype).pin_memory()
    output = torch.empty_like(host, pin_memory=True)
    ranges = SimpleNamespace(qkv=[], q=[], kv=[], ffn=[])

    def project(tile, start, stop):
        ranges.qkv.append((start, stop))
        q, k, v = F.linear(tile, weights[0]).split(256, dim=-1)
        return tuple(x.view(stop - start, 4, 64) for x in (q, k, v))

    def project_q(tile, destination, start, stop):
        ranges.q.append((start, stop))
        destination.copy_(F.linear(tile, weights[0][:256]).view(stop - start, 4, 64))

    def project_kv(tile, k, v, start, stop):
        ranges.kv.append((start, stop))
        out = F.linear(tile, weights[0][256:]).view(stop - start, 2, 4, 64)
        k.copy_(out[:, 0])
        v.copy_(out[:, 1])

    def epilogue(attention, residual, start, stop):
        update = F.linear(attention, weights[1])
        return residual[start:stop].to(device, non_blocking=True).add_(update)

    def ffn(post, start, stop):
        ranges.ffn.append((start, stop))
        normalized = F.rms_norm(post, (32,))
        gate, value = F.linear(normalized, weights[2]).chunk(2, dim=-1)
        return post.add_(F.linear(F.silu(gate) * value, weights[3]))

    ops = H3BlockOps(attention_epilogue=epilogue, ffn=ffn)
    meta = H3SequenceMeta(torch.tensor([0, 96, 161], dtype=torch.int32))
    if mode == "materialized":
        runner.run_block_(host, meta, H3MaterializedProjection(project), ops)
        actual = host
        attention = runner.projected_attention.attention._workspace
        staging = runner.projected_attention._projection_workspace.hidden
    else:
        runner.run_block(host, output, meta, H3RecomputeProjection(project_q, project_kv), ops)
        actual = output
        attention = runner.recomputed_attention.attention._workspace
        staging = [runner.recomputed_attention.workspace.hidden]
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    estimate = build_h3_block_execution(
        shape,
        config,
        profile(),
        callbacks=H3CallbackConfig(variant="block25", linear_memory="dense"),
    )
    assert ranges.ffn == estimate.metadata["ffn_ranges"]
    if mode == "materialized":
        assert ranges.qkv == estimate.metadata["projection_ranges"]
    else:
        assert ranges.q == [(start, stop) for start, stop, _ in estimate.metadata["query_ranges"]]
        assert ranges.kv == [(start, stop) for _, start, stop, _ in estimate.metadata["kv_ranges"]]
    tensors = [
        attention.q,
        attention.running_max,
        attention.running_sum,
        attention.accumulator,
        *attention.k,
        *attention.v,
        *attention.output,
        *staging,
        runner.workspace.carry,
        *runner.workspace.final_output,
    ]
    assert (
        sum(t.numel() * t.element_size() for t in tensors)
        == estimate.metadata["core_persistent_cuda_bytes"]
    )
    assert runner.plan.estimated_workspace_bytes == estimate.metadata["core_workspace_budget_bytes"]


@torch.inference_mode()
def test_cuda_operator_sample_measures_extra_workspace_and_separates_output():
    from seqattn_core.estimation import measure_cuda_h3_operator

    x = torch.ones(1024, 128, device="cuda")

    def operation():
        scratch = torch.ones(1024, 1024, device="cuda")
        output = x.clone()
        output.add_(scratch[:, :128])
        return output

    measured = measure_cuda_h3_operator(
        operation,
        tokens=1024,
        output_allocation_bytes=x.numel() * x.element_size(),
        warmup=1,
        repeats=3,
        provenance="synthetic workspace calibration",
    )
    assert len(measured.seconds) == 3
    assert measured.sample.extra_workspace_bytes >= 1024 * 1024 * 4
    assert measured.sample.seconds > 0


@pytest.mark.parametrize("tokens,channel_scale", [(8192, False), (8192, True), (16384, True)])
@torch.inference_mode()
def test_int8_workspace_model_matches_selected_eager_implementation(tokens, channel_scale):
    eager = pytest.importorskip("comfy_kitchen.backends.eager.quantization")
    from seqattn_core.estimation import measure_cuda_h3_operator
    from seqattn_core.estimation.h3.linear_memory import eager_int8_workspace

    inputs, outputs = 5376, 28672
    x = torch.randn(tokens, inputs, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (outputs, inputs), device="cuda", dtype=torch.int8)
    scales = torch.full(
        (outputs if channel_scale else 1,), 0.01, device="cuda", dtype=torch.float32
    )
    measured = measure_cuda_h3_operator(
        lambda: eager.int8_linear(x, weight, scales, convrot=True, convrot_groupsize=256),
        tokens=tokens,
        output_allocation_bytes=tokens * outputs * 2,
        warmup=1,
        repeats=2,
        provenance="current local eager INT8 backend, synthetic weights at H3 FC1 shape",
    )
    predicted = eager_int8_workspace(tokens, inputs, outputs, 2, per_channel_scale=channel_scale)
    assert abs(predicted - measured.sample.extra_workspace_bytes) <= 2 * 2**20, (
        predicted,
        measured.sample.extra_workspace_bytes,
    )


@pytest.mark.parametrize("tokens", [1, 4096])
@torch.inference_mode()
def test_eager_swiglu_fc2_workspace_matches_real_input_act_path(tokens):
    eager = pytest.importorskip("comfy_kitchen.backends.eager.quantization")
    from seqattn_core.estimation import measure_cuda_h3_operator
    from seqattn_core.estimation.h3.linear_memory import eager_int8_swiglu_workspace

    inputs, outputs = 14336, 5376
    x = torch.randn(tokens, 2 * inputs, device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (outputs, inputs), device="cuda", dtype=torch.int8)
    scales = torch.full((outputs,), 0.01, device="cuda", dtype=torch.float32)
    measurement = measure_cuda_h3_operator(
        lambda: eager.int8_linear(x, weight, scales, convrot=True, input_act="swiglu"),
        tokens=tokens,
        output_allocation_bytes=tokens * outputs * 2,
        warmup=1,
        repeats=2,
        provenance="actual eager input_act SwiGLU/FC2 path at H3 shape",
    )
    predicted = eager_int8_swiglu_workspace(tokens, inputs, outputs, 2, per_channel_scale=True)
    assert abs(predicted - measurement.sample.extra_workspace_bytes) <= 2 * 2**20
