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
from seqattn_core.dit.common import TiledStageOp
from seqattn_core.dit.ltx2 import (
    LTX2AttentionOps,
    LTX2BlockOps,
    LTX2MaterializedProjections,
    LTX2MaterializedRunner,
    LTX2RecomputeProjections,
    LTX2RecomputeRunner,
    LTX2SequenceMeta,
)
from seqattn_core.kernels import triton_is_available
from seqattn_core.projection import (
    CrossProjection,
    CrossRecomputeProjection,
    SelfProjection,
    SelfRecomputeProjection,
)


def _attention(q, k, v, q_bounds, k_bounds, scale):
    output = torch.empty_like(q)
    repeat = q.shape[1] // k.shape[1]
    for (q_start, q_stop), (k_start, k_stop) in zip(pairwise(q_bounds), pairwise(k_bounds)):
        expanded_k = k[k_start:k_stop].repeat_interleave(repeat, dim=1)
        expanded_v = v[k_start:k_stop].repeat_interleave(repeat, dim=1)
        result = torch.nn.functional.scaled_dot_product_attention(
            q[q_start:q_stop].transpose(0, 1).unsqueeze(0),
            expanded_k.transpose(0, 1).unsqueeze(0),
            expanded_v.transpose(0, 1).unsqueeze(0),
            scale=scale,
        )
        output[q_start:q_stop].copy_(result.squeeze(0).transpose(0, 1))
    return output


def _linear(in_features, out_features, dtype):
    return torch.nn.Linear(in_features, out_features, bias=False).to("cuda", dtype)


def _build_modules(video_features, audio_features, text_features, heads, kv_heads, dim, dtype):
    inner = heads * dim
    kv_inner = kv_heads * dim
    return {
        "video_self_qkv": _linear(video_features, 3 * inner, dtype),
        "video_self_out": _linear(inner, video_features, dtype),
        "audio_self_qkv": _linear(audio_features, 3 * inner, dtype),
        "audio_self_out": _linear(inner, audio_features, dtype),
        "video_text_q": _linear(video_features, inner, dtype),
        "video_text_kv": _linear(text_features, 2 * kv_inner, dtype),
        "video_text_out": _linear(inner, video_features, dtype),
        "audio_text_q": _linear(audio_features, inner, dtype),
        "audio_text_kv": _linear(text_features, 2 * kv_inner, dtype),
        "audio_text_out": _linear(inner, audio_features, dtype),
        "video_audio_q": _linear(video_features, inner, dtype),
        "video_audio_kv": _linear(audio_features, 2 * kv_inner, dtype),
        "video_audio_out": _linear(inner, video_features, dtype),
        "audio_video_q": _linear(audio_features, inner, dtype),
        "audio_video_kv": _linear(video_features, 2 * kv_inner, dtype),
        "audio_video_out": _linear(inner, audio_features, dtype),
        "video_ffn_in": _linear(video_features, 2 * 48, dtype),
        "video_ffn_out": _linear(48, video_features, dtype),
        "audio_ffn_in": _linear(audio_features, 2 * 40, dtype),
        "audio_ffn_out": _linear(40, audio_features, dtype),
    }


def _self_stage(hidden, qkv_linear, out_linear, bounds, heads, dim):
    qkv = qkv_linear(hidden).view(hidden.shape[0], 3, heads, dim)
    attended = _attention(qkv[:, 0], qkv[:, 1], qkv[:, 2], bounds, bounds, dim**-0.5)
    return hidden + out_linear(attended.reshape(hidden.shape[0], -1))


def _cross_stage(query, context, q_linear, kv_linear, out_linear, q_bounds, kv_bounds, shape):
    heads, kv_heads, dim = shape
    q = q_linear(query).view(query.shape[0], heads, dim)
    kv = kv_linear(context).view(context.shape[0], 2, kv_heads, dim)
    attended = _attention(q, kv[:, 0], kv[:, 1], q_bounds, kv_bounds, dim**-0.5)
    return query + out_linear(attended.reshape(query.shape[0], -1))


def _ffn(hidden, input_linear, output_linear):
    gate, up = input_linear(hidden).chunk(2, dim=-1)
    return hidden + output_linear(torch.nn.functional.silu(gate) * up)


def _reference_post_text(video, audio, text, modules, bounds, shape):
    heads, _, dim = shape
    video_bounds, audio_bounds, text_bounds = bounds
    video = _self_stage(
        video,
        modules["video_self_qkv"],
        modules["video_self_out"],
        video_bounds,
        heads,
        dim,
    )
    audio = _self_stage(
        audio,
        modules["audio_self_qkv"],
        modules["audio_self_out"],
        audio_bounds,
        heads,
        dim,
    )
    video = _cross_stage(
        video,
        text,
        modules["video_text_q"],
        modules["video_text_kv"],
        modules["video_text_out"],
        video_bounds,
        text_bounds,
        shape,
    )
    audio = _cross_stage(
        audio,
        text,
        modules["audio_text_q"],
        modules["audio_text_kv"],
        modules["audio_text_out"],
        audio_bounds,
        text_bounds,
        shape,
    )
    return video, audio


def _reference(video, audio, text, modules, bounds, shape):
    video_bounds, audio_bounds, _ = bounds
    video_snapshot, audio_snapshot = _reference_post_text(
        video, audio, text, modules, bounds, shape
    )
    video = _cross_stage(
        video_snapshot,
        audio_snapshot,
        modules["video_audio_q"],
        modules["video_audio_kv"],
        modules["video_audio_out"],
        video_bounds,
        audio_bounds,
        shape,
    )
    audio = _cross_stage(
        audio_snapshot,
        video_snapshot,
        modules["audio_video_q"],
        modules["audio_video_kv"],
        modules["audio_video_out"],
        audio_bounds,
        video_bounds,
        shape,
    )
    return (
        _ffn(video, modules["video_ffn_in"], modules["video_ffn_out"]),
        _ffn(audio, modules["audio_ffn_in"], modules["audio_ffn_out"]),
    )


def _config():
    return StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=16,
        kv_chunk_tokens=11,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )


def _runtime_config(plan):
    return StreamingAttentionConfig(
        backend="triton",
        workspace_budget_bytes=plan.workspace_budget_bytes,
        num_kv_buffers=plan.num_kv_buffers,
        num_output_buffers=plan.num_output_buffers,
        output_mode=plan.output_mode,
    )


def _self_runner(tokens, heads, dim, dtype):
    config = _config()
    plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    return ProjectedAttentionRunner(
        plan,
        _runtime_config(plan),
        ProjectionPipelineConfig(projection_tile_tokens=13),
    )


def _recompute_self_runner(tokens, hidden_features, heads, dim, dtype):
    config = _config()
    plan = build_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=config,
    )
    return RecomputedAttentionRunner(
        plan,
        hidden_features=hidden_features,
        attention_config=_runtime_config(plan),
    )


def _recompute_cross_runner(
    q_tokens,
    kv_tokens,
    query_features,
    context_features,
    heads,
    kv_heads,
    dim,
    dtype,
):
    config = _config()
    plan = build_plan(
        q_heads=heads,
        kv_heads=kv_heads,
        head_dim=dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=q_tokens,
        max_kv_tokens=kv_tokens,
        config=config,
    )
    return RecomputedCrossAttentionRunner(
        plan,
        query_hidden_features=query_features,
        context_hidden_features=context_features,
        attention_config=_runtime_config(plan),
    )


def _cross_runner(q_tokens, kv_tokens, heads, kv_heads, dim, dtype):
    config = _config()
    plan = build_plan(
        q_heads=heads,
        kv_heads=kv_heads,
        head_dim=dim,
        dtype=dtype,
        device="cuda",
        max_q_tokens=q_tokens,
        max_kv_tokens=kv_tokens,
        config=config,
    )
    return ProjectedCrossAttentionRunner(
        plan,
        _runtime_config(plan),
        ProjectionPipelineConfig(projection_tile_tokens=13),
    )


def _capture(captures, name, tile, start, stop):
    captures.setdefault(name, []).append((start, stop, tile.detach().cpu()))


def _captured(captures, name):
    return torch.cat(
        [tile for _, _, tile in sorted(captures[name], key=lambda item: item[0])], dim=0
    )


def test_ltx2_padding_masks_accept_prefixes_and_reject_additive_masks():
    meta = LTX2SequenceMeta.from_padding_masks(
        torch.tensor([[True, True, False], [True, False, False]]),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[True, True, True, False], [True, False, False, False]]),
    )
    meta.validate(3, 3, 4)
    torch.testing.assert_close(meta.video_cu_seqlens, torch.tensor([0, 2, 3], dtype=torch.int32))

    with pytest.raises(ValueError, match="additive QxK masks"):
        LTX2SequenceMeta.from_padding_masks(
            torch.zeros((2, 3), dtype=torch.float32),
            torch.ones((2, 2), dtype=torch.bool),
            torch.ones((2, 4), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="valid prefix"):
        LTX2SequenceMeta.from_padding_masks(
            torch.tensor([[True, False, True]]),
            torch.ones((1, 2), dtype=torch.bool),
            torch.ones((1, 4), dtype=torch.bool),
        )


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_ltx2_materialized_block_matches_reference_and_preserves_cross_snapshot():
    torch.manual_seed(701)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    video_tokens, audio_tokens, text_tokens = 43, 29, 19
    video_features, audio_features, text_features = 32, 24, 20
    heads, kv_heads, dim = 2, 1, 16
    video_bounds = [0, 17, video_tokens]
    audio_bounds = [0, 11, audio_tokens]
    text_bounds = [0, 7, text_tokens]
    video = torch.randn(video_tokens, video_features, dtype=dtype, pin_memory=True)
    audio = torch.randn(audio_tokens, audio_features, dtype=dtype, pin_memory=True)
    text = torch.randn(text_tokens, text_features, dtype=dtype, pin_memory=True)
    original_video = video.clone()
    original_audio = audio.clone()
    modules = _build_modules(
        video_features,
        audio_features,
        text_features,
        heads,
        kv_heads,
        dim,
        dtype,
    )
    runner = LTX2MaterializedRunner(
        video_self_attention=_self_runner(video_tokens, heads, dim, dtype),
        audio_self_attention=_self_runner(audio_tokens, heads, dim, dtype),
        video_text_attention=_cross_runner(video_tokens, text_tokens, heads, kv_heads, dim, dtype),
        audio_text_attention=_cross_runner(audio_tokens, text_tokens, heads, kv_heads, dim, dtype),
        video_from_audio_attention=_cross_runner(
            video_tokens, audio_tokens, heads, kv_heads, dim, dtype
        ),
        audio_from_video_attention=_cross_runner(
            audio_tokens, video_tokens, heads, kv_heads, dim, dtype
        ),
        video_hidden_features=video_features,
        audio_hidden_features=audio_features,
        video_ffn_tile_tokens=17,
        audio_ffn_tile_tokens=13,
    )
    captures = {}

    def self_projection(name):
        linear = modules[f"{name}_self_qkv"]

        def project(tile, start, stop):
            del start, stop
            qkv = linear(tile).view(-1, 3, heads, dim)
            return tuple(qkv[:, index].contiguous() for index in range(3))

        return SelfProjection(project)

    def cross_projection(prefix, query_capture=None, context_capture=None):
        q_linear = modules[f"{prefix}_q"]
        kv_linear = modules[f"{prefix}_kv"]

        def project_q(tile, start, stop):
            if query_capture is not None:
                _capture(captures, query_capture, tile, start, stop)
            return q_linear(tile).view(-1, heads, dim)

        def project_kv(tile, start, stop):
            if context_capture is not None:
                _capture(captures, context_capture, tile, start, stop)
            kv = kv_linear(tile).view(-1, 2, kv_heads, dim)
            return kv[:, 0].contiguous(), kv[:, 1].contiguous()

        return CrossProjection(project_q, project_kv)

    def attention_ops(output_name):
        output = modules[output_name]

        def epilogue(attention, residual_host, start, stop):
            return output(attention) + residual_host[start:stop].to(device, non_blocking=True)

        return LTX2AttentionOps(epilogue)

    def ffn_op(prefix):
        input_linear = modules[f"{prefix}_ffn_in"]
        output_linear = modules[f"{prefix}_ffn_out"]

        def operation(tile, start, stop):
            del start, stop
            gate, up = input_linear(tile).chunk(2, dim=-1)
            return tile + output_linear(torch.nn.functional.silu(gate) * up)

        return TiledStageOp(operation)

    projections = LTX2MaterializedProjections(
        self_projection("video"),
        self_projection("audio"),
        cross_projection("video_text"),
        cross_projection("audio_text"),
        cross_projection("video_audio", "video_as_query", "audio_as_context"),
        cross_projection("audio_video", "audio_as_query", "video_as_context"),
    )
    ops = LTX2BlockOps(
        attention_ops("video_self_out"),
        attention_ops("audio_self_out"),
        attention_ops("video_text_out"),
        attention_ops("audio_text_out"),
        attention_ops("video_audio_out"),
        attention_ops("audio_video_out"),
        ffn_op("video"),
        ffn_op("audio"),
    )
    runner.run_block_(
        video,
        audio,
        text,
        LTX2SequenceMeta(
            torch.tensor(video_bounds, dtype=torch.int32),
            torch.tensor(audio_bounds, dtype=torch.int32),
            torch.tensor(text_bounds, dtype=torch.int32),
        ),
        projections,
        ops,
    )
    expected_video, expected_audio = _reference(
        original_video.to(device),
        original_audio.to(device),
        text.to(device),
        modules,
        (video_bounds, audio_bounds, text_bounds),
        (heads, kv_heads, dim),
    )

    torch.testing.assert_close(video, expected_video.cpu(), atol=9e-2, rtol=1e-2)
    torch.testing.assert_close(audio, expected_audio.cpu(), atol=9e-2, rtol=1e-2)
    torch.testing.assert_close(
        _captured(captures, "video_as_query"),
        _captured(captures, "video_as_context"),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _captured(captures, "audio_as_query"),
        _captured(captures, "audio_as_context"),
        atol=0,
        rtol=0,
    )


def _captured_snapshot(captures, name, tokens):
    result = None
    covered = torch.zeros(tokens, dtype=torch.bool)
    for start, stop, tile in captures[name]:
        if result is None:
            result = torch.empty((tokens, tile.shape[1]), dtype=tile.dtype)
        overlap = covered[start:stop]
        if overlap.any():
            torch.testing.assert_close(result[start:stop][overlap], tile[overlap], atol=0, rtol=0)
        result[start:stop].copy_(tile)
        covered[start:stop] = True
    assert result is not None and covered.all()
    return result


@pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")
@torch.inference_mode()
def test_ltx2_recompute_matches_reference_preserves_snapshots_and_ping_pongs():
    torch.manual_seed(709)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    video_tokens, audio_tokens, text_tokens = 43, 29, 19
    video_features, audio_features, text_features = 32, 24, 20
    heads, kv_heads, dim = 2, 1, 16
    video_bounds = [0, 17, video_tokens]
    audio_bounds = [0, 11, audio_tokens]
    text_bounds = [0, 7, text_tokens]
    bounds = (video_bounds, audio_bounds, text_bounds)
    shape = (heads, kv_heads, dim)
    video_initial = torch.randn(video_tokens, video_features, dtype=dtype, pin_memory=True)
    audio_initial = torch.randn(audio_tokens, audio_features, dtype=dtype, pin_memory=True)
    text = torch.randn(text_tokens, text_features, dtype=dtype, pin_memory=True)
    modules = _build_modules(
        video_features,
        audio_features,
        text_features,
        heads,
        kv_heads,
        dim,
        dtype,
    )
    runner = LTX2RecomputeRunner(
        video_self_attention=_recompute_self_runner(
            video_tokens, video_features, heads, dim, dtype
        ),
        audio_self_attention=_recompute_self_runner(
            audio_tokens, audio_features, heads, dim, dtype
        ),
        video_text_attention=_recompute_cross_runner(
            video_tokens,
            text_tokens,
            video_features,
            text_features,
            heads,
            kv_heads,
            dim,
            dtype,
        ),
        audio_text_attention=_recompute_cross_runner(
            audio_tokens,
            text_tokens,
            audio_features,
            text_features,
            heads,
            kv_heads,
            dim,
            dtype,
        ),
        video_from_audio_attention=_recompute_cross_runner(
            video_tokens,
            audio_tokens,
            video_features,
            audio_features,
            heads,
            kv_heads,
            dim,
            dtype,
        ),
        audio_from_video_attention=_recompute_cross_runner(
            audio_tokens,
            video_tokens,
            audio_features,
            video_features,
            heads,
            kv_heads,
            dim,
            dtype,
        ),
        video_ffn_tile_tokens=17,
        audio_ffn_tile_tokens=13,
    )
    captures = {}

    def self_projection(name):
        linear = modules[f"{name}_self_qkv"]

        def project_q(tile, destination, start, stop):
            del start, stop
            qkv = linear(tile).view(-1, 3, heads, dim)
            destination.copy_(qkv[:, 0])

        def project_kv(tile, destination_k, destination_v, start, stop):
            del start, stop
            qkv = linear(tile).view(-1, 3, heads, dim)
            destination_k.copy_(qkv[:, 1])
            destination_v.copy_(qkv[:, 2])

        return SelfRecomputeProjection(project_q, project_kv)

    def cross_projection(prefix, query_capture=None, context_capture=None):
        q_linear = modules[f"{prefix}_q"]
        kv_linear = modules[f"{prefix}_kv"]

        def project_q(tile, destination, start, stop):
            if query_capture is not None:
                _capture(captures, query_capture, tile, start, stop)
            destination.copy_(q_linear(tile).view(-1, heads, dim))

        def project_kv(tile, destination_k, destination_v, start, stop):
            if context_capture is not None:
                _capture(captures, context_capture, tile, start, stop)
            kv = kv_linear(tile).view(-1, 2, kv_heads, dim)
            destination_k.copy_(kv[:, 0])
            destination_v.copy_(kv[:, 1])

        return CrossRecomputeProjection(project_q, project_kv)

    def attention_ops(output_name):
        output = modules[output_name]

        def epilogue(attention, residual_host, start, stop):
            return output(attention) + residual_host[start:stop].to(device, non_blocking=True)

        return LTX2AttentionOps(epilogue)

    def ffn_op(prefix):
        input_linear = modules[f"{prefix}_ffn_in"]
        output_linear = modules[f"{prefix}_ffn_out"]

        def operation(tile, start, stop):
            del start, stop
            gate, up = input_linear(tile).chunk(2, dim=-1)
            return tile + output_linear(torch.nn.functional.silu(gate) * up)

        return TiledStageOp(operation)

    projections = LTX2RecomputeProjections(
        self_projection("video"),
        self_projection("audio"),
        cross_projection("video_text"),
        cross_projection("audio_text"),
        cross_projection("video_audio", "video_as_query", "audio_as_context"),
        cross_projection("audio_video", "audio_as_query", "video_as_context"),
    )
    ops = LTX2BlockOps(
        attention_ops("video_self_out"),
        attention_ops("audio_self_out"),
        attention_ops("video_text_out"),
        attention_ops("audio_text_out"),
        attention_ops("video_audio_out"),
        attention_ops("audio_video_out"),
        ffn_op("video"),
        ffn_op("audio"),
    )
    meta = LTX2SequenceMeta(
        torch.tensor(video_bounds, dtype=torch.int32),
        torch.tensor(audio_bounds, dtype=torch.int32),
        torch.tensor(text_bounds, dtype=torch.int32),
    )
    video_source = video_initial.clone().pin_memory()
    audio_source = audio_initial.clone().pin_memory()
    video_destination = torch.empty_like(video_source, pin_memory=True)
    audio_destination = torch.empty_like(audio_source, pin_memory=True)
    result_video, result_audio = runner.run_block(
        video_source,
        video_destination,
        audio_source,
        audio_destination,
        text,
        meta,
        projections,
        ops,
    )
    post_text_video, post_text_audio = _reference_post_text(
        video_initial.to(device),
        audio_initial.to(device),
        text.to(device),
        modules,
        bounds,
        shape,
    )
    expected_video, expected_audio = _reference(
        video_initial.to(device),
        audio_initial.to(device),
        text.to(device),
        modules,
        bounds,
        shape,
    )

    assert result_video.data_ptr() == video_destination.data_ptr()
    assert result_audio.data_ptr() == audio_destination.data_ptr()
    torch.testing.assert_close(video_source, post_text_video.cpu(), atol=9e-2, rtol=1e-2)
    torch.testing.assert_close(audio_source, post_text_audio.cpu(), atol=9e-2, rtol=1e-2)
    torch.testing.assert_close(result_video, expected_video.cpu(), atol=1e-1, rtol=2e-2)
    torch.testing.assert_close(result_audio, expected_audio.cpu(), atol=1e-1, rtol=2e-2)
    torch.testing.assert_close(
        _captured_snapshot(captures, "video_as_query", video_tokens),
        _captured_snapshot(captures, "video_as_context", video_tokens),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _captured_snapshot(captures, "audio_as_query", audio_tokens),
        _captured_snapshot(captures, "audio_as_context", audio_tokens),
        atol=0,
        rtol=0,
    )
    attention_runners = (
        runner.video_self_attention,
        runner.audio_self_attention,
        runner.video_text_attention,
        runner.audio_text_attention,
        runner.video_from_audio_attention,
        runner.audio_from_video_attention,
    )
    assert not any(
        hasattr(attention_runner, name)
        for attention_runner in attention_runners
        for name in ("q_cpu", "k_cpu", "v_cpu", "arena")
    )

    one_video = video_initial.clone().pin_memory()
    one_audio = audio_initial.clone().pin_memory()
    one_video_scratch = torch.empty_like(one_video, pin_memory=True)
    one_audio_scratch = torch.empty_like(one_audio, pin_memory=True)
    odd_video, odd_audio = runner.run_blocks_(
        one_video,
        one_video_scratch,
        one_audio,
        one_audio_scratch,
        text,
        meta,
        [(projections, ops)],
    )
    assert odd_video.data_ptr() == one_video_scratch.data_ptr()
    assert odd_audio.data_ptr() == one_audio_scratch.data_ptr()

    two_video = video_initial.clone().pin_memory()
    two_audio = audio_initial.clone().pin_memory()
    two_video_scratch = torch.empty_like(two_video, pin_memory=True)
    two_audio_scratch = torch.empty_like(two_audio, pin_memory=True)
    even_video, even_audio = runner.run_blocks_(
        two_video,
        two_video_scratch,
        two_audio,
        two_audio_scratch,
        text,
        meta,
        [(projections, ops), (projections, ops)],
    )
    expected_two_video, expected_two_audio = _reference(
        expected_video,
        expected_audio,
        text.to(device),
        modules,
        bounds,
        shape,
    )
    assert even_video.data_ptr() == two_video.data_ptr()
    assert even_audio.data_ptr() == two_audio.data_ptr()
    torch.testing.assert_close(even_video, expected_two_video.cpu(), atol=1.2e-1, rtol=2e-2)
    torch.testing.assert_close(even_audio, expected_two_audio.cpu(), atol=1.2e-1, rtol=2e-2)
