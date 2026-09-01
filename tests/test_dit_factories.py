from types import SimpleNamespace

import pytest
import torch

from seqattn_core import StreamingAttentionConfig, build_plan
from seqattn_core.dit.ltx2 import LTX2AttentionPlans, LTX2Config, build_ltx2_runner
from seqattn_core.dit.minimax_h3 import H3Config, build_h3_runner
from seqattn_core.dit.wan import WanAttentionPlans, WanConfig, build_wan_runner
from seqattn_core.projection import MaterializedQKVArena


def _recording_constructor(name, calls):
    def construct(*args, **kwargs):
        instance = SimpleNamespace(kind=name, args=args, kwargs=kwargs)
        calls.append(instance)
        return instance

    return construct


def _patch_constructors(monkeypatch, module, names):
    calls = []
    for name in names:
        monkeypatch.setattr(module, name, _recording_constructor(name, calls))
    return calls


def test_h3_factory_selects_mode_and_propagates_tiles(monkeypatch):
    from seqattn_core.dit.minimax_h3 import factory

    calls = _patch_constructors(
        monkeypatch,
        factory,
        (
            "ProjectedAttentionRunner",
            "RecomputedAttentionRunner",
            "H3MaterializedRunner",
            "H3RecomputeRunner",
        ),
    )
    plan = object()
    materialized = build_h3_runner(
        plan,
        hidden_features=96,
        config=H3Config(
            execution_mode="materialized",
            projection_tile_tokens=1536,
            ffn_tile_tokens=768,
        ),
    )
    assert materialized.kind == "H3MaterializedRunner"
    projected = next(call for call in calls if call.kind == "ProjectedAttentionRunner")
    assert projected.args[0] is plan
    assert projected.args[2].projection_tile_tokens == 1536
    assert materialized.kwargs["ffn_tile_tokens"] == 768

    calls.clear()
    recompute = build_h3_runner(
        plan,
        hidden_features=80,
        config=H3Config(execution_mode="recompute", ffn_tile_tokens=640),
    )
    assert recompute.kind == "H3RecomputeRunner"
    producer = next(call for call in calls if call.kind == "RecomputedAttentionRunner")
    assert producer.args[0] is plan
    assert producer.kwargs["hidden_features"] == 80
    assert recompute.kwargs["ffn_tile_tokens"] == 640


def test_wan_factory_selects_mode_and_shares_one_materialized_arena(monkeypatch):
    from seqattn_core.dit.wan import factory

    calls = _patch_constructors(
        monkeypatch,
        factory,
        (
            "ProjectedAttentionRunner",
            "ProjectedCrossAttentionRunner",
            "RecomputedAttentionRunner",
            "RecomputedCrossAttentionRunner",
            "WanMaterializedRunner",
            "WanRecomputeRunner",
        ),
    )
    arena_calls = []

    class FakeArena:
        @classmethod
        def for_plans(cls, plans, *, pin_memory):
            arena = SimpleNamespace(plans=tuple(plans), pin_memory=pin_memory)
            arena_calls.append(arena)
            return arena

    monkeypatch.setattr(factory, "MaterializedQKVArena", FakeArena)
    plans = WanAttentionPlans(object(), object())
    materialized = build_wan_runner(
        plans,
        hidden_features=72,
        text_hidden_features=48,
        config=WanConfig(
            execution_mode="materialized",
            projection_tile_tokens=1152,
            ffn_tile_tokens=576,
        ),
    )
    assert materialized.kind == "WanMaterializedRunner"
    assert len(arena_calls) == 1
    producers = [
        call
        for call in calls
        if call.kind in {"ProjectedAttentionRunner", "ProjectedCrossAttentionRunner"}
    ]
    assert len(producers) == 2
    assert {id(call.kwargs["arena"]) for call in producers} == {id(arena_calls[0])}
    assert all(call.args[2].projection_tile_tokens == 1152 for call in producers)
    assert materialized.kwargs["ffn_tile_tokens"] == 576

    calls.clear()
    arena_calls.clear()
    recompute = build_wan_runner(
        plans,
        hidden_features=72,
        text_hidden_features=48,
        config=WanConfig(execution_mode="recompute", ffn_tile_tokens=512),
    )
    assert recompute.kind == "WanRecomputeRunner"
    assert not arena_calls
    cross = next(call for call in calls if call.kind == "RecomputedCrossAttentionRunner")
    assert cross.kwargs["query_hidden_features"] == 72
    assert cross.kwargs["context_hidden_features"] == 48
    assert recompute.kwargs["ffn_tile_tokens"] == 512


def test_ltx2_factory_selects_mode_and_shares_two_materialized_arenas(monkeypatch):
    from seqattn_core.dit.ltx2 import factory

    calls = _patch_constructors(
        monkeypatch,
        factory,
        (
            "ProjectedAttentionRunner",
            "ProjectedCrossAttentionRunner",
            "RecomputedAttentionRunner",
            "RecomputedCrossAttentionRunner",
            "LTX2MaterializedRunner",
            "LTX2RecomputeRunner",
        ),
    )
    arena_calls = []

    class FakeArena:
        @classmethod
        def for_plans(cls, plans, *, pin_memory):
            arena = SimpleNamespace(plans=tuple(plans), pin_memory=pin_memory)
            arena_calls.append(arena)
            return arena

    monkeypatch.setattr(factory, "MaterializedQKVArena", FakeArena)
    plan_values = [object() for _ in range(6)]
    plans = LTX2AttentionPlans(*plan_values)
    materialized = build_ltx2_runner(
        plans,
        video_hidden_features=96,
        audio_hidden_features=64,
        text_hidden_features=40,
        config=LTX2Config(
            execution_mode="materialized",
            projection_tile_tokens=1280,
            video_ffn_tile_tokens=704,
            audio_ffn_tile_tokens=448,
        ),
    )
    assert materialized.kind == "LTX2MaterializedRunner"
    assert len(arena_calls) == 2
    arena_by_plan = {id(plan): arena for arena in arena_calls for plan in arena.plans}
    producers = [
        call
        for call in calls
        if call.kind in {"ProjectedAttentionRunner", "ProjectedCrossAttentionRunner"}
    ]
    assert len(producers) == 6
    assert all(call.kwargs["arena"] is arena_by_plan[id(call.args[0])] for call in producers)
    assert all(call.args[2].projection_tile_tokens == 1280 for call in producers)
    assert materialized.kwargs["video_ffn_tile_tokens"] == 704
    assert materialized.kwargs["audio_ffn_tile_tokens"] == 448

    calls.clear()
    arena_calls.clear()
    recompute = build_ltx2_runner(
        plans,
        video_hidden_features=96,
        audio_hidden_features=64,
        text_hidden_features=40,
        config=LTX2Config(
            execution_mode="recompute",
            video_ffn_tile_tokens=640,
            audio_ffn_tile_tokens=384,
        ),
    )
    assert recompute.kind == "LTX2RecomputeRunner"
    assert not arena_calls
    crosses = [call for call in calls if call.kind == "RecomputedCrossAttentionRunner"]
    assert [
        (call.kwargs["query_hidden_features"], call.kwargs["context_hidden_features"])
        for call in crosses
    ] == [
        (96, 40),
        (64, 40),
        (96, 64),
        (64, 96),
    ]
    assert recompute.kwargs["video_ffn_tile_tokens"] == 640
    assert recompute.kwargs["audio_ffn_tile_tokens"] == 384


def _plan(*, q_tokens, kv_tokens, q_heads=2, kv_heads=1, dtype=torch.float32):
    return build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=16,
        dtype=dtype,
        device="cpu",
        max_q_tokens=q_tokens,
        max_kv_tokens=kv_tokens,
        config=StreamingAttentionConfig(backend="reference"),
    )


def test_materialized_qkv_arena_shares_compatible_capacity_and_reports_allocation():
    first = _plan(q_tokens=7, kv_tokens=5)
    second = _plan(q_tokens=3, kv_tokens=11)
    arena = MaterializedQKVArena.for_plans((first, second), pin_memory=False)

    assert arena.max_q_tokens == 7
    assert arena.max_kv_tokens == 11
    q, k, v = arena.views(3, 4)
    assert q.shape == (3, 2, 16)
    assert k.shape == v.shape == (4, 1, 16)
    element_size = torch.empty((), dtype=torch.float32).element_size()
    assert arena.allocated_bytes == (7 * 2 * 16 + 2 * 11 * 1 * 16) * element_size
    assert arena.allocated_bytes > sum(t.numel() * t.element_size() for t in (q, k, v))


@pytest.mark.parametrize(
    "incompatible",
    [
        _plan(q_tokens=7, kv_tokens=5, q_heads=4, kv_heads=1),
        _plan(q_tokens=7, kv_tokens=5, dtype=torch.float16),
    ],
)
def test_materialized_qkv_arena_rejects_incompatible_layouts(incompatible):
    compatible = _plan(q_tokens=7, kv_tokens=5)
    with pytest.raises(ValueError, match="matching heads, head_dim, and dtype"):
        MaterializedQKVArena.for_plans((compatible, incompatible), pin_memory=False)
