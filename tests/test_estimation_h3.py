from contextlib import nullcontext
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from seqattn_core import StreamingAttentionConfig, build_attention_plan
from seqattn_core.dit.minimax_h3.consumer import H3DeviceOutputConsumer
from seqattn_core.dit.minimax_h3.types import (
    estimate_h3_materialized_aux_workspace_bytes,
    estimate_h3_recompute_aux_workspace_bytes,
)
from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3OperatorProfile,
    H3OperatorSample,
    H3WeightPolicy,
    MemoryPool,
    RateProfile,
    build_h3_block_execution,
    memory_statistics,
    schedule_execution,
)


def profile():
    compute = RateProfile("analytical compute", 1e10, ("matrix",))
    return H3DeviceProfile.from_rates(
        "synthetic accelerator",
        device_pool=MemoryPool("HBM"),
        host_pool=MemoryPool("DRAM"),
        attention=compute,
        gemm=compute,
        vector=RateProfile("vector", 1e9, ("vector",)),
        h2d=RateProfile("H2D", 1e8, ("copy.in",), kind="io"),
        d2h=RateProfile("D2H", 1e8, ("copy.out",), kind="io"),
        d2d=RateProfile("D2D", 1e9, ("copy.local",)),
    )


def build(shape=None, config=None, callbacks=None, **kwargs):
    return build_h3_block_execution(
        shape or H3BlockShape((9, 5), 32, 64, 4, 8, rope_dim=8),
        config or H3ExecutionConfig(4, 3, 4, 8),
        profile(),
        callbacks=callbacks or H3CallbackConfig(variant="block25", linear_memory="dense"),
        **kwargs,
    )


class Event:
    def record(self, *args):
        pass

    def wait_event(self, *args):
        pass


def actual_consumer_ranges(monkeypatch, lengths, q_chunk, ffn_chunk):
    """Run the real consumer's state machine with CPU tensors and dummy CUDA events."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: Event())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    workspace = SimpleNamespace(
        hidden_features=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        ffn_tile_tokens=ffn_chunk,
        final_output_chunk_tokens=ffn_chunk,
        carry=torch.empty(ffn_chunk, 2),
        final_output=[torch.empty(ffn_chunk, 2) for _ in range(2)],
        output_pending=[False, False],
        output_ready=[Event(), Event()],
        output_free=[Event(), Event()],
        d2h_stream=Event(),
    )
    calls = []

    def ffn(tile, start, stop):
        # Ensure carried ranges contain the right rows, not just the right count.
        torch.testing.assert_close(tile[:, 0], torch.arange(start, stop, dtype=torch.float32))
        calls.append((start, stop))
        return tile

    stats = SimpleNamespace(ffn_tiles=0, final_hidden_d2h_bytes=0, ffn_cross_q_boundaries=0)
    consumer = H3DeviceOutputConsumer(workspace)
    hidden = torch.zeros(sum(lengths), 2)
    consumer.reset(
        destination_hidden_host=hidden,
        residual_hidden_host=hidden,
        ops=SimpleNamespace(attention_epilogue=lambda a, *_: a, ffn=ffn),
        stats=stats,
    )
    offset = 0
    for length in lengths:
        for start in range(offset, offset + length, q_chunk):
            stop = min(start + q_chunk, offset + length)
            tile = torch.arange(start, stop, dtype=torch.float32)[:, None].expand(-1, 2).clone()
            consumer(tile, start, stop)
        offset += length
    consumer.finish()
    return calls, stats.ffn_cross_q_boundaries


@pytest.mark.parametrize(
    "q,ffn,lengths",
    [(4, 8, (9, 5)), (7, 3, (5, 11)), (4, 4, (4, 8)), (16, 24, (13, 24)), (3, 11, (2, 2, 3))],
)
def test_h3_ranges_match_real_consumer_including_carry_tail_and_segments(
    monkeypatch, q, ffn, lengths
):
    expected, crossings = actual_consumer_ranges(monkeypatch, lengths, q, ffn)
    spec = build(
        shape=H3BlockShape(lengths, 2, 8, 1, 8, rope_dim=8), config=H3ExecutionConfig(q, 3, 4, ffn)
    )
    assert spec.metadata["ffn_ranges"] == expected
    assert spec.metadata["ffn_cross_q_boundaries"] == crossings
    assert spec.metadata["operator_counts"]["fc1"] == len(expected)
    schedule_execution(spec)


@pytest.mark.parametrize(
    "ffn,expected", [(2048, 40), (4096, 20), (8192, 10), (12288, 7), (16384, 5)]
)
def test_measured_block25_ffn_call_counts(ffn, expected):
    spec = build(
        shape=H3BlockShape((81159,), 5376, 14336, 56, 128),
        config=H3ExecutionConfig(3840, 4096, 4096, ffn),
    )
    assert len(spec.metadata["ffn_ranges"]) == expected
    assert spec.metadata["operator_counts"]["attention"] == 440
    assert spec.metadata["operator_counts"]["qkv"] == 20


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
def test_persistent_workspace_matches_runtime_estimators(mode):
    shape = H3BlockShape((64,), 128, 256, 4, 64, rope_dim=32)
    c = H3ExecutionConfig(32, 16, 20, 48, execution_mode=mode)
    spec = build(shape, c)
    plan = build_attention_plan(
        q_heads=4,
        kv_heads=4,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cpu",
        max_q_tokens=64,
        max_kv_tokens=64,
        config=StreamingAttentionConfig(
            q_chunk_tokens=32,
            kv_chunk_tokens=16,
            block_m=16,
            block_n=16,
            output_mode="device_consumer",
        ),
    )
    if mode == "materialized":
        aux = estimate_h3_materialized_aux_workspace_bytes(
            hidden_features=128,
            dtype=torch.bfloat16,
            projection_tile_tokens=20,
            num_projection_buffers=2,
            ffn_tile_tokens=48,
        )
    else:
        aux = estimate_h3_recompute_aux_workspace_bytes(
            hidden_features=128, dtype=torch.bfloat16, hidden_staging_tokens=32, ffn_tile_tokens=48
        )
    assert spec.metadata["core_workspace_budget_bytes"] == plan.estimated_workspace_bytes + aux
    core_buffers = [
        b for b in spec.buffers if b.owner == "operator" and b.persistent and b.pool == "HBM"
    ]
    assert sum(b.size_bytes for b in core_buffers) == spec.metadata["core_persistent_cuda_bytes"]
    assert not any(b.component == "reserve" for b in spec.buffers)


def test_host_inplace_and_recompute_distinct_sources():
    mat = build()
    rec = build(config=H3ExecutionConfig(4, 3, 4, 8, execution_mode="recompute"))
    assert "hidden.output" not in {b.name for b in mat.buffers}
    assert {"hidden.source", "hidden.output"} <= {b.name for b in rec.buffers}
    assert not any(b.name.startswith("host.Q") for b in rec.buffers)
    for spec in (mat, rec):
        trace = schedule_execution(spec)
        assert all(p.active_peak_bytes <= p.peak_bytes for p in memory_statistics(trace).pools)


def test_producer_global_ranges_and_attention_segment_boundaries():
    spec = build(
        shape=H3BlockShape((2, 2), 32, 64, 4, 8, rope_dim=8), config=H3ExecutionConfig(4, 4, 4, 3)
    )
    assert spec.metadata["operator_counts"]["qkv"] == 1
    assert spec.metadata["query_ranges"] == [(0, 2, 0), (2, 4, 1)]
    assert spec.metadata["ffn_ranges"] == [(0, 3), (3, 4)]
    trace = schedule_execution(spec)
    by_name = {op.name: op for op in trace.operations}
    barrier = by_name["projection.global_kv_barrier"].end_seconds
    assert all(
        op.start_seconds >= barrier for op in trace.operations if op.component == "attention"
    )
    result = next(b for b in trace.buffers if b.name == "projection0.qkv.result")
    assert result.end_seconds == barrier


def test_recompute_q_kv_projection_and_actual_staging_capacities():
    c = H3ExecutionConfig(4, 3, 4, 8, execution_mode="recompute")
    spec = build(config=c)
    assert spec.metadata["operator_counts"]["q"] == 5
    assert spec.metadata["operator_counts"]["kv"] == 13
    assert next(b.size_bytes for b in spec.buffers if b.name == "recompute.hidden") == 4 * 32 * 2
    assert any(op.name.endswith(".write_1") for op in spec.operations)
    other = build(config=replace(c, projection_tile_tokens=99))
    assert spec.operations == other.operations
    assert spec.buffers == other.buffers


def test_complete_modulated_callbacks_and_no_same_stream_overlap():
    spec = build(callbacks=H3CallbackConfig(variant="modulated", linear_memory="dense"))
    counts = spec.metadata["operator_counts"]
    for key in (
        "adaln",
        "norm1",
        "modulate1",
        "rope_angles",
        "rope_table",
        "qk_rope",
        "attention",
        "finalize",
        "out",
        "attention_residual",
        "norm2",
        "modulate2",
        "fc1",
        "swiglu",
        "fc2",
        "ffn_residual",
    ):
        assert counts[key] > 0
    trace = schedule_execution(spec)
    compute = [e for e in trace.operations if "stream=compute;" in e.details]
    assert all(a.end_seconds <= b.start_seconds for a, b in pairwise(compute))
    assert any(b.component == "RoPE table" for b in trace.buffers)


def test_output_reuse_waits_for_d2h_and_parent_post_survives_view_consumption():
    spec = build(
        shape=H3BlockShape((17,), 32, 64, 4, 8, rope_dim=8),
        config=H3ExecutionConfig(12, 8, 4, 3, num_output_buffers=1),
    )
    trace = schedule_execution(spec)
    events = {e.name: e for e in trace.operations}
    assert events["ffn1.norm2"].start_seconds >= events["ffn0.output_d2h"].end_seconds
    post = next(b for b in trace.buffers if b.name == "query0.post_attention")
    assert post.size_bytes == 12 * 32 * 2
    assert post.end_seconds == events["query0.consumer_return"].end_seconds
    assert not any("view" in b.name for b in trace.buffers)


def test_weight_phase_lifetimes_and_owner_filtered_activation_statistics():
    spec = build(weights=H3WeightPolicy(mode="staged", projection_bytes=4000, consumer_bytes=5000))
    trace = schedule_execution(spec)
    buffers = {b.name: b for b in trace.buffers}
    assert buffers["weights.projection"].end_seconds <= buffers["weights.consumer"].start_seconds
    total = memory_statistics(trace).pools[0].peak_bytes
    activations = (
        memory_statistics(trace, owners=frozenset({"operator", "callback", "caller"}))
        .pools[0]
        .peak_bytes
    )
    assert total >= activations
    rec = schedule_execution(
        build(
            config=H3ExecutionConfig(4, 3, 4, 8, execution_mode="recompute"),
            weights=H3WeightPolicy(mode="staged", projection_bytes=4000, consumer_bytes=5000),
        )
    )
    weights = [b for b in rec.buffers if b.owner == "weights" and b.pool == "HBM"]
    assert all(b.end_seconds == rec.duration_seconds for b in weights)


def test_operator_samples_are_shape_exact_and_override_analytical_workspace():
    p = profile()
    sample = H3OperatorProfile(
        samples=(H3OperatorSample(8, 0.25, 10000), H3OperatorSample(6, 0.2, 9000)),
        provenance="measured FC1",
    )
    p = replace(p, operators={**p.operators, "fc1": sample}).for_shape(
        H3BlockShape((9, 5), 32, 64, 4, 8, rope_dim=8)
    )
    spec = build_h3_block_execution(
        H3BlockShape((9, 5), 32, 64, 4, 8, rope_dim=8),
        H3ExecutionConfig(4, 3, 4, 8),
        p,
        callbacks=H3CallbackConfig(variant="block25", linear_memory="dense"),
    )
    assert next(op.duration_seconds for op in spec.operations if op.name == "ffn0.fc1") == 0.25
    assert next(b.size_bytes for b in spec.buffers if b.name == "ffn0.fc1.workspace") == 10000
    with pytest.raises(ValueError, match="no calibrated operator sample"):
        build_h3_block_execution(
            H3BlockShape((9,), 32, 64, 4, 8, rope_dim=8),
            H3ExecutionConfig(4, 3, 4, 4),
            p,
            callbacks=H3CallbackConfig(variant="block25"),
        )


def test_int8_intermediates_scale_with_actual_ffn_tile_not_q_cap():
    shape = H3BlockShape((20000,), 5376, 14336, 56, 128)
    peaks = []
    for ffn in (8192, 16384):
        spec = build(
            shape,
            H3ExecutionConfig(3840, 4096, 4096, ffn),
            H3CallbackConfig(variant="block25", linear_memory="int8_eager"),
        )
        scratch = next(b for b in spec.buffers if b.name == "ffn0.fc1.workspace")
        assert scratch.size_bytes > ffn * 14336 * 4
        peaks.append(memory_statistics(schedule_execution(spec)).pools[0].peak_bytes)
    assert peaks[1] - peaks[0] > 1500 * 2**20


def test_local_pool_workspace_and_profile_json_roundtrip():
    import json

    from seqattn_core.estimation import H3ScratchBuffer

    p = profile()
    p = replace(
        p,
        local_pools=(MemoryPool("NPU.core0.UB", 4096),),
        operators={
            **p.operators,
            "fc1": replace(
                p.operators["fc1"],
                additional_workspaces=(H3ScratchBuffer("NPU.core0.UB", bytes_per_token=1024),),
            ),
        },
    )
    restored = H3DeviceProfile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored == p
    spec = build_h3_block_execution(
        H3BlockShape((9, 5), 32, 64, 4, 8, rope_dim=8),
        H3ExecutionConfig(4, 3, 4, 8),
        restored,
        callbacks=H3CallbackConfig(variant="block25"),
    )
    stats = memory_statistics(schedule_execution(spec))
    local = next(pool for pool in stats.pools if pool.pool == "NPU.core0.UB")
    assert local.peak_bytes == 8192
    assert local.exceeds_capacity


def test_sample_signature_rejects_wrong_head_or_dtype():
    shape = H3BlockShape((9, 5), 32, 64, 4, 8, rope_dim=8)
    p = profile().for_shape(shape)
    with pytest.raises(ValueError, match="signature"):
        build_h3_block_execution(
            replace(shape, heads=8),
            H3ExecutionConfig(4, 3, 4, 8),
            p,
            callbacks=H3CallbackConfig(variant="block25"),
        )


def test_transfer_samples_distinguish_equal_token_counts_with_different_payloads():
    p = profile()
    samples = (H3OperatorSample(4, 0.1, 0, work=128), H3OperatorSample(4, 0.2, 0, work=256))
    transfer = replace(p.operators["h2d"], samples=samples)
    assert transfer.resolve(4, 128)[0] == 0.1
    assert transfer.resolve(4, 256)[0] == 0.2
    with pytest.raises(ValueError, match="payload_bytes"):
        replace(
            p,
            operators={
                **p.operators,
                "h2d": H3OperatorProfile(samples=(H3OperatorSample(4, 0.1, 0),)),
            },
        )


def test_runtime_plan_binding_preserves_distinct_host_arena_capacities():
    plan = build_attention_plan(
        q_heads=4,
        kv_heads=4,
        head_dim=64,
        dtype=torch.bfloat16,
        device="cpu",
        max_q_tokens=512,
        max_kv_tokens=1024,
        config=StreamingAttentionConfig(
            q_chunk_tokens=64, kv_chunk_tokens=64, output_mode="device_consumer"
        ),
    )
    config = H3ExecutionConfig.from_attention_plan(
        plan, projection_tile_tokens=48, ffn_tile_tokens=80
    )
    spec = build(H3BlockShape((161,), 32, 64, 4, 64, rope_dim=32), config)
    assert next(b.size_bytes for b in spec.buffers if b.name == "host.Q") == 512 * 4 * 64 * 2
    assert next(b.size_bytes for b in spec.buffers if b.name == "host.K") == 1024 * 4 * 64 * 2


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_projection_overlap_preserves_slot_reuse_and_global_barrier(slots):
    from seqattn_core.estimation.web.inputs import prepare_request

    _, shape, configs, device, callbacks, weights, _ = prepare_request(
        {"tokens": 16384, "num_projection_buffers": slots}
    )
    spec = build_h3_block_execution(shape, configs[0], device, callbacks=callbacks, weights=weights)
    trace = schedule_execution(spec)
    events = {o.name: o for o in trace.operations}
    assert spec.scheduling_policy == "earliest_ready"
    for index in range(slots, 4):
        assert (
            events[f"projection{index}.hidden_h2d"].start_seconds
            >= events[f"projection{index - slots}.V.d2h"].end_seconds
        )
    if slots > 1:
        gemm = events["projection1.qkv"]
        copy = events["projection0.Q.d2h"]
        assert gemm.start_seconds == events["projection1.modulate1"].end_seconds
        assert min(gemm.end_seconds, copy.end_seconds) > max(gemm.start_seconds, copy.start_seconds)
        # Packing still competes with GEMM; no fake concurrent compute.
        assert events["projection0.K.pack"].start_seconds >= gemm.end_seconds
    else:
        assert events["projection1.qkv"].start_seconds >= events["projection0.V.d2h"].end_seconds
    barrier = events["projection.global_kv_barrier"].end_seconds
    assert barrier >= max(events[f"projection{i}.V.d2h"].end_seconds for i in range(4))
    assert events["query0.q_h2d"].start_seconds >= barrier
    # Validate every stream/resource and the modeled buffer lifetime invariants.
    from seqattn_core.estimation import trace_from_measurements

    trace_from_measurements(
        spec,
        {o.name: (o.start_seconds, o.end_seconds) for o in trace.operations},
        provenance="synthetic H3 overlap regression",
    )
