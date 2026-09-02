# SeqAttn 0.4 execution architecture

Status: implemented on the `0.4.0a1` feature branch on 2026-09-02.

## Objective

Maximize reuse of projection and attention execution mechanisms while keeping
model control flow direct and readable. SeqAttn does not use a generic block
program, graph interpreter, logical stream router, or model-name dispatch.

The governing rule is:

> Reuse execution mechanisms, not model control flow.

Wan, LTX2, and H3 runners express their real block order as ordinary Python
calls. Consumer callbacks continue to own normalization, projections, residual
updates, FFNs, and weight leases.

## Resolved attention plans

`StreamingAttentionConfig` is a request type. `build_attention_plan()` consumes
it once and produces an immutable `AttentionPlan` containing:

- validated shape and capacity limits;
- aligned Q/KV chunks and kernel launch parameters;
- workspace estimates and the optional workspace budget;
- resolved backend request, pinning policy, output mode, and NVTX policy.

`StreamingAttentionRunner` accepts only the resolved plan. There is no runtime
config reconciliation and no second source of truth.

Automatic Q sizing is deterministic: when a workspace budget is present, the
builder selects the largest aligned resident-Q chunk that fits. The default KV
chunk is a conservative 8192-token cap aligned to the selected kernel. The plan
builder does not contain nominal PCIe bandwidth, nominal GPU throughput, or a
pseudo-performance cost model. Performance defaults must come from measured
calibration.

## Projection mechanisms

The projection package has seven focused modules:

```text
projection/
  __init__.py
  api.py             functional entry points
  contracts.py       callback and lease contracts
  materialized.py    shared asynchronous materialization producer
  recomputed.py      shared direct-write executor plus self/cross facades
  runners.py         materialized self/cross runner facades
  memory.py          host QKV arena and persistent CUDA staging
  validation.py      hidden and projected tensor contracts
```

Materialized self- and cross-attention facades share one asynchronous producer loop for
H2D staging, projection callbacks, validation, D2H copies, slot reuse, recovery,
and statistics. Exactness still requires a complete host K/V readiness barrier
before attention consumption.

Recomputed self- and cross-attention share one execution base and the same
streaming attention tile-source contract. Their public workspaces remain
explicit because query and context staging have different capacities and are
part of failure-recovery and memory-accounting tests.

## DiT mechanisms

`dit/common/attention.py` exposes two policy-specific executors:

- `MaterializedAttentionExecutor` materializes self/cross QKV and consumes a
  completed batch through a caller-selected output consumer;
- `RecomputedAttentionExecutor` projects Q and K/V directly into attention
  tiles and writes through a caller-selected output consumer.

These executors contain no model topology. Model runners retain explicit order:

- Wan: self-attention, text cross-attention, FFN;
- LTX2: video/audio self-attention, video/audio text attention, bidirectional
  audio/video attention, video/audio FFNs;
- H3: its dedicated fused attention/FFN consumer.

LTX2 materialized mode makes snapshot safety visible in code by materializing
both bidirectional cross-attention directions before consuming either result.
LTX2 recompute mode makes routing visible through named source and destination
buffers. No generic snapshot descriptor or buffer router is needed.

## Module rules

- Prefer four to eight cohesive modules per implementation directory.
- Treat 150-350 lines as a normal implementation size.
- Review files above 500 lines for mixed responsibilities, but do not split a
  cohesive algorithm only to satisfy a line count.
- Keep public callback types stable and model-neutral.
- Keep materialized and recompute policy code separate when their ownership or
  lifetime rules differ.
- Remove compatibility facades during the 0.4 alpha API break rather than
  carrying duplicate paths indefinitely.

## Preserved invariants

- packed `cu_seqlens` remain hard segment boundaries;
- exact attention observes all K/V for a segment before finalizing Q;
- online softmax state remains FP32 and bounded by resident Q;
- persistent runners and workspaces remain single-flight;
- asynchronous paths preserve explicit pinned-memory requirements;
- projection callback failures synchronize/recover persistent staging before
  runner reuse;
- H3 integration remains callback-driven and model agnostic.
