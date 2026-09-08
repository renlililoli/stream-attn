"""Full single-GPU-style H3 dense block, described without executing a device."""

from __future__ import annotations

from .callbacks import H3Callbacks
from .consumer import H3Consumer
from .graph import H3Graph
from .profiles import H3WeightPolicy


def _allocate(g):
    s, c = g.shape, g.config
    h, a, size = s.hidden_features, s.attention_features, s.element_bytes
    g.allocate(
        "hidden.source",
        s.tokens * h * size,
        "host hidden",
        owner="caller",
        persistent=True,
        host=True,
    )
    if c.execution_mode == "recompute":
        g.allocate(
            "hidden.output",
            s.tokens * h * size,
            "host output",
            owner="caller",
            persistent=True,
            host=True,
        )
    g.allocate(
        "Q", c.q_chunk_tokens * a * size, "Q / attention output", owner="operator", persistent=True
    )
    g.allocate(
        "accumulator",
        c.q_chunk_tokens * a * 4,
        "FP32 accumulator",
        owner="operator",
        persistent=True,
    )
    for name in ("running_max", "running_sum"):
        g.allocate(
            name,
            c.q_chunk_tokens * s.heads * 4,
            "FP32 softmax scalars",
            owner="operator",
            persistent=True,
        )
    for slot in range(c.num_kv_buffers):
        for tensor in ("K", "V"):
            g.allocate(
                f"{tensor}.{slot}",
                c.kv_tile_tokens * a * size,
                tensor,
                owner="operator",
                persistent=True,
            )
    g.allocate(
        "ffn.carry", c.ffn_tile_tokens * h * size, "FFN carry", owner="operator", persistent=True
    )
    for slot in range(c.num_output_buffers):
        g.allocate(
            f"output.{slot}",
            c.ffn_tile_tokens * h * size,
            "output slots",
            owner="operator",
            persistent=True,
        )
    if c.execution_mode == "materialized":
        q_capacity = c.qkv_capacity_tokens or s.tokens
        kv_capacity = c.kv_capacity_tokens or q_capacity
        if min(q_capacity, kv_capacity) < s.tokens:
            raise ValueError("QKV arena capacity is smaller than the input")
        for tensor in ("Q", "K", "V"):
            capacity = q_capacity if tensor == "Q" else kv_capacity
            g.allocate(
                f"host.{tensor}",
                capacity * a * size,
                f"host {tensor}",
                owner="operator",
                persistent=True,
                host=True,
            )
        for slot in range(c.num_projection_buffers):
            g.allocate(
                f"projection.hidden.{slot}",
                c.projection_tile_tokens * h * size,
                "projection hidden staging",
                owner="operator",
                persistent=True,
            )
    else:
        g.allocate(
            "recompute.hidden",
            max(c.q_chunk_tokens, c.kv_tile_tokens) * h * size,
            "recompute hidden staging",
            owner="operator",
            persistent=True,
        )


def _weight_group(g, group, *, dependencies=()):
    size = getattr(g.weights, f"{group}_bytes")
    if not size:
        return None, None
    name = g.allocate(
        f"weights.{group}",
        size,
        f"{group} weights",
        owner="weights",
        persistent=g.weights.mode == "resident",
    )
    if g.weights.mode == "resident":
        return name, None
    host = g.allocate(
        f"host.weights.{group}", size, "host weights", owner="weights", persistent=True, host=True
    )
    ready = g.copy(
        f"weights.{group}.h2d",
        "h2d",
        1,
        size,
        host,
        name,
        dependencies=dependencies,
        stream="weights.h2d",
        component="weight transfer",
    )
    return name, ready


def _materialize(g, callbacks, projection_ready):
    s, c = g.shape, g.config
    slot_done = [None] * c.num_projection_buffers
    slot_results = [None] * c.num_projection_buffers
    gate = None
    previous_result = None
    for index, start in enumerate(range(0, s.tokens, c.projection_tile_tokens)):
        stop = min(start + c.projection_tile_tokens, s.tokens)
        tokens, slot = stop - start, index % c.num_projection_buffers
        prefix = f"projection{index}"
        # _run_tiles synchronizes the reused slot on the host before submitting
        # new H2D. Independent slots can overlap; the producer is not fully serial.
        gate = g.milestone(f"{prefix}.slot_released", (gate, slot_done[slot]))
        if slot_results[slot]:
            g.retain(slot_results[slot], gate)
        staging = f"projection.hidden.{slot}"
        copied = g.copy(
            f"{prefix}.hidden_h2d",
            "h2d",
            tokens,
            tokens * s.hidden_features * s.element_bytes,
            "hidden.source",
            staging,
            dependencies=(gate,),
            stream="projection.h2d",
            component="projection hidden transfer",
        )
        g.milestone(f"{prefix}.input_ready", (copied, projection_ready), stream="compute")
        result, projected = callbacks.project(prefix, staging, tokens, "qkv")
        # With one slot the loop's previous local `projected` also survives the
        # release of keepalive until the next callback has supplied its result.
        if c.num_projection_buffers == 1 and previous_result:
            g.retain(previous_result, projected)
        previous_result = result
        done = projected
        for tensor in ("Q", "K", "V"):
            source = result
            if g.callbacks.qkv_result_layout == "strided":
                source = g.allocate(
                    f"{prefix}.{tensor}.packed",
                    tokens * s.attention_features * s.element_bytes,
                    "QKV D2H packing",
                    owner="operator",
                )
                done = g.copy(
                    f"{prefix}.{tensor}.pack",
                    "d2d",
                    tokens,
                    tokens * s.attention_features * s.element_bytes,
                    result,
                    source,
                    dependencies=(projected,),
                    stream="projection.d2h",
                )
            done = g.copy(
                f"{prefix}.{tensor}.d2h",
                "d2h",
                tokens,
                tokens * s.attention_features * s.element_bytes,
                source,
                f"host.{tensor}",
                dependencies=(done,),
                stream="projection.d2h",
                component="QKV transfer",
            )
        slot_done[slot], slot_results[slot] = done, result
    barrier = g.milestone("projection.global_kv_barrier", slot_done)
    for result in slot_results:
        g.retain(result, barrier)
    return barrier


def _attention(g, callbacks, consumer, ready):
    s, c = g.shape, g.config
    kv_free = [None] * c.num_kv_buffers
    q_free = hidden_free = None
    offset = q_index = 0
    q_ranges, kv_ranges = [], []
    for segment, length in enumerate(s.segments):
        for q_start in range(offset, offset + length, c.q_chunk_tokens):
            q_stop = min(q_start + c.q_chunk_tokens, offset + length)
            tokens = q_stop - q_start
            prefix = f"query{q_index}"
            q_ranges.append((q_start, q_stop, segment))
            if c.execution_mode == "materialized":
                copied = g.copy(
                    f"{prefix}.q_h2d",
                    "h2d",
                    tokens,
                    tokens * s.attention_features * s.element_bytes,
                    "host.Q",
                    "Q",
                    dependencies=(ready, q_free),
                    stream="attention.h2d",
                    component="Q transfer",
                )
                g.milestone(f"{prefix}.q_ready", (copied,), stream="compute")
            else:
                copied = g.copy(
                    f"{prefix}.hidden_h2d",
                    "h2d",
                    tokens,
                    tokens * s.hidden_features * s.element_bytes,
                    "hidden.source",
                    "recompute.hidden",
                    dependencies=(ready, hidden_free),
                    stream="recompute.h2d",
                    component="recompute hidden transfer",
                )
                g.milestone(f"{prefix}.hidden_ready", (copied,), stream="compute")
                _, hidden_free = callbacks.project(
                    prefix, "recompute.hidden", tokens, "q", destinations=("Q",)
                )
            for index, start in enumerate(range(offset, offset + length, c.kv_tile_tokens)):
                stop = min(start + c.kv_tile_tokens, offset + length)
                kv_tokens, slot = stop - start, index % c.num_kv_buffers
                kv_ranges.append((q_index, start, stop, segment))
                kv_prefix = f"{prefix}.kv{index}"
                destinations = (f"K.{slot}", f"V.{slot}")
                if c.execution_mode == "materialized":
                    copied = None
                    for tensor, destination in zip(("K", "V"), destinations):
                        copied = g.copy(
                            f"{kv_prefix}.{tensor}.h2d",
                            "h2d",
                            kv_tokens,
                            kv_tokens * s.attention_features * s.element_bytes,
                            f"host.{tensor}",
                            destination,
                            dependencies=(ready, kv_free[slot]),
                            stream="attention.h2d",
                            component="KV transfer",
                        )
                    g.milestone(f"{kv_prefix}.ready", (copied,), stream="compute")
                else:
                    copied = g.copy(
                        f"{kv_prefix}.hidden_h2d",
                        "h2d",
                        kv_tokens,
                        kv_tokens * s.hidden_features * s.element_bytes,
                        "hidden.source",
                        "recompute.hidden",
                        dependencies=(hidden_free,),
                        stream="recompute.h2d",
                        component="recompute hidden transfer",
                    )
                    g.milestone(f"{kv_prefix}.hidden_ready", (copied,), stream="compute")
                    _, hidden_free = callbacks.project(
                        kv_prefix, "recompute.hidden", kv_tokens, "kv", destinations=destinations
                    )
                updated = g.op(
                    f"{kv_prefix}.update",
                    "attention",
                    tokens,
                    4 * tokens * kv_tokens * s.attention_features,
                    ("Q", *destinations, "accumulator", "running_max", "running_sum"),
                    kv_tokens=kv_tokens,
                    details=f"Q=[{q_start},{q_stop}); KV=[{start},{stop}); segment={segment}; initialize={index == 0}",
                )
                kv_free[slot] = updated
            g.op(
                f"{prefix}.finalize",
                "finalize",
                tokens,
                tokens * s.attention_features,
                ("accumulator", "running_sum", "Q"),
                details="attention output aliases resident Q",
            )
            q_free = consumer.consume(prefix, q_start, q_stop)
            q_index += 1
        offset += length
    return q_ranges, kv_ranges


def build_h3_block_execution(shape, config, profile, *, callbacks, weights=None, name=None):
    """Model the actual dense H3 runner + explicitly selected callback implementation.

    No model loader, CUDA query or framework adapter is imported. Operator samples
    or declared rate/workspace models provide hardware-specific costs. The fixed
    runner control flow, buffer sharing and context lifetimes come from core H3.
    """
    weights = H3WeightPolicy() if weights is None else weights
    profile.validate_shape(shape)
    if config.q_chunk_tokens % profile.q_alignment or config.kv_tile_tokens % profile.kv_alignment:
        raise ValueError("attention capacities must be aligned to the selected operator profile")
    if callbacks.variant == "modulated" and shape.rope_dim <= 0:
        raise ValueError("modulated callbacks require a positive RoPE dimension")
    g = H3Graph(shape, config, profile, callbacks, weights)
    _allocate(g)
    auxiliary_weights, auxiliary_ready = _weight_group(g, "auxiliary")
    if auxiliary_ready:
        g.milestone("auxiliary.weights_ready", (auxiliary_ready,), stream="compute")
    cb = H3Callbacks(g)
    cb.prepare()
    projection_weights, projection_ready = _weight_group(g, "projection")
    if config.execution_mode == "materialized":
        barrier = _materialize(g, cb, projection_ready)
        if projection_weights:
            g.retain(projection_weights, barrier)
        consumer_weights, consumer_ready = _weight_group(g, "consumer", dependencies=(barrier,))
        ready = g.milestone("attention.begin", (barrier, consumer_ready), stream="compute")
    else:
        consumer_weights, consumer_ready = _weight_group(g, "consumer")
        ready = g.milestone("attention.begin", (projection_ready, consumer_ready), stream="compute")
    consumer = H3Consumer(g, cb)
    q_ranges, kv_ranges = _attention(g, cb, consumer, ready)
    done = consumer.finish()
    for buffer in (consumer_weights, auxiliary_weights, cb.modulation):
        g.retain(buffer, done)
    if config.execution_mode == "recompute":
        g.retain(projection_weights, done)
    # Weight owners are independent from activation owners in memory_statistics.
    for op in g.operations:
        if op.component in {"qkv", "q", "kv"}:
            g.use((projection_weights,), op.name)
        if op.component in {"out", "fc1", "fc2", "swiglu_fc2"}:
            g.use((consumer_weights,), op.name)
        if op.component in {"norm1", "norm2", "qk_rope", "qk_norm", "adaln"}:
            g.use((auxiliary_weights,), op.name)
    spec = g.finish(
        name
        or f"H3 {config.execution_mode} · Q={config.q_chunk_tokens} · KV={config.kv_tile_tokens} · FFN={config.ffn_tile_tokens}"
    )
    spec.metadata.update(
        projection_ranges=[
            (start, min(start + config.projection_tile_tokens, shape.tokens))
            for start in range(0, shape.tokens, config.projection_tile_tokens)
        ]
        if config.execution_mode == "materialized"
        else [],
        query_ranges=q_ranges,
        kv_ranges=kv_ranges,
        ffn_ranges=consumer.ffn_ranges,
        ffn_cross_q_boundaries=consumer.cross_q_boundaries,
    )
    return spec
