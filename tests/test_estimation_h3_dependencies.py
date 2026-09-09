"""Run the actual core executor/source/consumer state machines with recording CUDA events.

Tensor arithmetic is irrelevant here. Stream/event happens-before relationships
are compared with the estimator on common operations, independently of durations.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_estimation_h3 import build

from seqattn_core.dit.minimax_h3.consumer import H3DeviceOutputConsumer
from seqattn_core.dit.minimax_h3.stats import H3DiTStats
from seqattn_core.dit.minimax_h3.types import H3BlockOps
from seqattn_core.dit.minimax_h3.workspace import H3BlockWorkspace
from seqattn_core.estimation import H3BlockShape, H3ExecutionConfig
from seqattn_core.projection.memory import RecomputeWorkspace
from seqattn_core.stats import RecomputedAttentionStats, StreamingAttentionStats
from seqattn_core.streaming import executor
from seqattn_core.streaming.tasks import build_query_tasks
from seqattn_core.streaming.tile_source import HostQKVTileSource, RecomputedQKVTileSource
from seqattn_core.streaming.workspace import CudaWorkspace


class Recorder:
    def __init__(self):
        self.nodes = {}
        self.host = set()
        self.current = None

    def op(self, name):
        assert name not in self.nodes
        self.nodes[name] = self.current.tail | self.host
        self.current.tail = {name}

    def anonymous(self):
        self.op(f"anonymous{len(self.nodes)}")


class Stream:
    def __init__(self, recorder):
        self.recorder, self.tail = recorder, set()

    def wait_event(self, event):
        self.tail |= event.tail

    def synchronize(self):
        self.recorder.host |= self.tail


class Event:
    def __init__(self, recorder):
        self.recorder, self.tail = recorder, set()

    def record(self, stream=None):
        self.tail = set((stream or self.recorder.current).tail)

    def synchronize(self):
        self.recorder.host |= self.tail

    def elapsed_time(self, other):
        return 0.0


def ancestors(nodes):
    result = {}
    for name, dependencies in nodes.items():
        result[name] = set(dependencies)
        for dependency in dependencies:
            result[name] |= result[dependency]
    return result


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
@pytest.mark.parametrize("q,ffn,slots", [(4, 8, 1), (4, 8, 2), (7, 3, 2), (4, 4, 1)])
def test_actual_runtime_event_dependencies_equal_estimator(monkeypatch, mode, q, ffn, slots):
    rec = Recorder()
    rec.current = Stream(rec)

    @contextmanager
    def stream_context(stream):
        previous = rec.current
        rec.current = stream
        try:
            yield
        finally:
            rec.current = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: rec.current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: Stream(rec))
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: Event(rec))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    plan = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        q_chunk_tokens=q,
        kv_chunk_tokens=3,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        num_kv_buffers=slots,
        num_output_buffers=slots,
        output_mode="device_consumer",
        enable_nvtx=False,
        block_m=16,
        block_n=16,
        num_warps=4,
        num_stages=1,
    )
    workspace = CudaWorkspace(plan)
    block = H3BlockWorkspace(
        hidden_features=8,
        ffn_tile_tokens=ffn,
        dtype=plan.dtype,
        device=plan.device,
        num_final_output_buffers=slots,
    )
    hidden = torch.zeros(14, 8)
    output = hidden if mode == "materialized" else torch.empty_like(hidden)
    consumer = H3DeviceOutputConsumer(block)
    state = SimpleNamespace(query=-1, kv=-1, phase="q", ffn=-1)
    storage = lambda t: t.untyped_storage().data_ptr()
    original_copy = torch.Tensor.copy_

    def copied(destination, source, *a, **kw):
        ptr = storage(destination)
        prefix = f"query{state.query}"
        kv_prefix = f"{prefix}.kv{state.kv}"
        if ptr == storage(workspace.q):
            name = f"{prefix}.q_h2d" if mode == "materialized" else f"{prefix}.write_0"
        elif ptr in [storage(x) for x in workspace.k]:
            name = f"{kv_prefix}.K.h2d" if mode == "materialized" else f"{kv_prefix}.write_0"
        elif ptr in [storage(x) for x in workspace.v]:
            name = f"{kv_prefix}.V.h2d" if mode == "materialized" else f"{kv_prefix}.write_1"
        elif ptr in [storage(x) for x in block.final_output]:
            name = f"ffn{state.ffn}.output_copy"
        elif ptr == storage(output):
            name = f"ffn{state.ffn}.output_d2h"
        elif mode == "recompute" and ptr == storage(staging.hidden):
            name = f"{prefix if state.phase == 'q' else kv_prefix}.hidden_h2d"
        else:
            rec.anonymous()  # Carry copies still contribute stream dependencies.
            return original_copy(destination, source, *a, **kw)
        rec.op(name)
        return original_copy(destination, source, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "copy_", copied)

    def epilogue(attention, residual, start, stop):
        rec.op(f"query{state.query}.out")
        rec.op(f"query{state.query}.residual_h2d")
        return torch.zeros(stop - start, 8)

    def feed_forward(post, start, stop):
        state.ffn += 1
        rec.op(f"ffn{state.ffn}.norm2")
        return post

    consumer.reset(
        destination_hidden_host=output,
        residual_hidden_host=hidden,
        ops=H3BlockOps(epilogue, feed_forward),
        stats=H3DiTStats(),
    )
    if mode == "materialized":
        host_qkv = [torch.zeros(14, 1, 8) for _ in range(3)]
        source = HostQKVTileSource(*host_qkv, workspace, enable_nvtx=False)
    else:
        staging = RecomputeWorkspace(
            hidden_features=8, staging_tokens=max(q, 3), dtype=plan.dtype, device=plan.device
        )

        def project_q(tile, destination, start, stop):
            destination.copy_(torch.zeros(stop - start, 1, 8))

        def project_kv(tile, k, v, start, stop):
            k.copy_(torch.zeros(stop - start, 1, 8))
            v.copy_(torch.zeros(stop - start, 1, 8))

        source = RecomputedQKVTileSource(
            hidden,
            staging,
            project_q=project_q,
            project_kv=project_kv,
            stats=RecomputedAttentionStats(),
            enable_nvtx=False,
        )
    load_q, load_kv = source.load_q, source.load_kv

    def query(*a, **kw):
        state.query += 1
        state.kv, state.phase = -1, "q"
        return load_q(*a, **kw)

    def kv(*a, **kw):
        state.kv += 1
        state.phase = "kv"
        return load_kv(*a, **kw)

    source.load_q, source.load_kv = query, kv
    monkeypatch.setattr(
        executor,
        "update_attention_state",
        lambda *a, **kw: rec.op(f"query{state.query}.kv{state.kv}.update"),
    )
    monkeypatch.setattr(
        executor, "finalize_attention", lambda *a, **kw: rec.op(f"query{state.query}.finalize")
    )
    engine = executor.TritonExecutorMixin()
    engine.plan, engine._workspace = plan, workspace
    tasks = build_query_tasks([0, 9, 14], [0, 9, 14], q_chunk_tokens=q)
    engine._run_triton_from_source_once(
        source, tasks, 1.0, False, None, StreamingAttentionStats(), output_consumer=consumer
    )
    spec = build(
        shape=H3BlockShape(
            (9, 5), 8, 16, 1, 8, element_bytes=4, activation_dtype="float32", rope_dim=8
        ),
        config=H3ExecutionConfig(
            q, 3, 4, ffn, execution_mode=mode, num_kv_buffers=slots, num_output_buffers=slots
        ),
    )
    actual = ancestors(rec.nodes)
    predicted = ancestors({o.name: set(o.dependencies) for o in spec.operations})
    common = set(actual) & set(predicted)
    assert len(common) > 25
    for name in common:
        assert actual[name] & common == predicted[name] & common, name


@pytest.mark.parametrize("slots", [1, 2, 3])
@pytest.mark.parametrize("blocking_rope", [False, True])
def test_real_projection_producer_host_and_stream_dependencies(monkeypatch, slots, blocking_rope):
    from seqattn_core.config import ProjectionPipelineConfig
    from seqattn_core.estimation import H3CallbackConfig
    from seqattn_core.projection.materialized import MaterializedProjectionProducer
    from seqattn_core.projection.memory import MaterializedQKVArena
    from seqattn_core.stats import ProjectedAttentionStats

    rec = Recorder()
    rec.current = Stream(rec)

    @contextmanager
    def context(stream):
        previous, rec.current = rec.current, stream
        try:
            yield
        finally:
            rec.current = previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: rec.current)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: Stream(rec))
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: Event(rec))
    monkeypatch.setattr(torch.cuda, "stream", context)
    plan = SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, q_heads=1, kv_heads=1, head_dim=8
    )
    arena = MaterializedQKVArena(
        max_q_tokens=9,
        max_kv_tokens=9,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        dtype=plan.dtype,
        pin_memory=False,
    )
    producer = MaterializedProjectionProducer(
        plan,
        ProjectionPipelineConfig(projection_tile_tokens=4, num_projection_buffers=slots),
        arena,
    )
    state = SimpleNamespace(index=-1)
    original_copy = torch.Tensor.copy_
    hosts = {
        t.untyped_storage().data_ptr(): name for t, name in zip((arena.q, arena.k, arena.v), "QKV")
    }

    def copied(destination, source, *a, **kw):
        tensor = hosts.get(destination.untyped_storage().data_ptr())
        if tensor:
            rec.op(f"projection{state.index}.{tensor}.d2h")
        else:
            state.index += 1
            rec.op(f"projection{state.index}.hidden_h2d")
        return original_copy(destination, source, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "copy_", copied)

    def project(tile, start, stop):
        rec.op(f"projection{state.index}.qkv")
        if blocking_rope:
            rec.op(f"projection{state.index}.qk_rope.position_h2d")
            # Observed production .to(device) waits for this stream on the CPU.
            rec.current.synchronize()
        output = torch.zeros(stop - start, 24)
        return tuple(t.view(stop - start, 1, 8) for t in output.split(8, -1))

    producer.project_qkv(torch.zeros(9, 8), project, ProjectedAttentionStats())
    rec.op("projection.global_kv_barrier")
    spec = build(
        shape=H3BlockShape(
            (9,), 8, 16, 1, 8, element_bytes=4, activation_dtype="float32", rope_dim=8
        ),
        config=H3ExecutionConfig(4, 3, 4, 8, num_projection_buffers=slots),
        callbacks=H3CallbackConfig(
            variant="modulated" if blocking_rope else "block25", linear_memory="dense"
        ),
    )
    actual = ancestors(rec.nodes)
    predicted = ancestors({o.name: set(o.dependencies) for o in spec.operations})
    common = set(actual) & set(predicted)
    assert len(common) >= 16
    for name in common:
        assert actual[name] & common == predicted[name] & common, name
