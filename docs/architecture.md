# SeqAttn architecture

Status: current for `seqattn-core 0.4.0a1` on 2026-09-02.

SeqAttn is an inference-only attention runtime for workloads whose complete
Q/K/V working set does not fit in GPU memory. Dense execution is exact;
MiniMax-H3 also has an explicit approximate Sol-style mode. The core
package owns
generic planning, execution, storage, and callback orchestration. Model
loading, checkpoint conversion, block eviction, and UI integration remain
consumer responsibilities.

## Package boundary

```text
src/seqattn_core/    public API and implementation; no compat facades
  _config_file.py    shared TOML path, parsing, and scalar validation
  api.py             functional public API
  config.py          execution policy dataclasses
  plan.py            deterministic workspace and execution planning
  stats.py           statistics dataclasses
  reference.py       FP32 online-softmax CPU reference
  validation.py      host tensor and sequence validation
  quantization.py    per-token-group INT8 quantization
  sparse/            Sol plan, streamed runner, stats, and semantic reference
  streaming/         contiguous CPU-DRAM -> HBM execution
    backend.py       config loading, SM policy, and capability checks
    flash_backends.py explicit FA2/FA3/FA4 partial-forward adapters
    flash_split_executor.py shared FA streaming and FP32 combine schedule
    workspace.py     persistent CUDA buffers, streams, and events
    executor.py      built-in Triton copy/compute/output schedule
    runner.py        validation, reference dispatch, and public runner
    tile_source.py   host-materialized and device-recomputed Q/K/V loaders
  paged/             fixed-host-budget page runtime
    layout.py        page descriptors and tensor/KV layouts
    protocols.py     PageSource/PageSink reader/writer contracts
    memory.py        caller-owned memory source and sinks
    memory_budget.py host allocation accounting and enforcement
    cache.py         bounded two-region K/V cache
    simulation.py    in-memory NVMe timing model
    runtime/         orchestration, staging, I/O, reference, and Triton paths
  storage/           persistent aligned backing stores
    direct_io.py     O_DIRECT helpers and aligned bounce buffers
    records.py       on-disk Q/KV page record construction
    qkv_store.py     Q/KV manifest validation and page reads
    qkv_writer.py    streaming Q/KV store construction
    output.py        paged output writer and loader
  projection/        hidden-state projection attention pipeline
    api.py           functional convenience API
    contracts.py     projection callbacks and immutable lease descriptors
    materialized.py  shared asynchronous materialization producer
    runners.py       materialized self/cross runner facades
    recomputed.py    shared self/cross direct-write recompute runners
    memory.py        host QKV arena and persistent CUDA staging
    validation.py    hidden and projected tensor contracts
  dit/               model-specific fixed-order DiT integrations
    common/          internal consumers, tiled stages, masks, and validation
    minimax_h3/      H3 materialized and recompute runners
    wan/             Wan materialized and recompute block runners
    ltx2/            LTX2 materialized video/audio block runner
  benchmarking/      repository benchmark tools; excluded from release wheels
    common.py        JSON, sequence bounds, RSS, and NVML sampling
    streaming.py     DRAM-backed and full-GPU benchmark
    paged.py         memory, simulated-NVMe, and NVMe benchmark
    projection.py    projected pipeline benchmark
  kernels/           Triton kernels and launch helpers
    sol_preprocess.py KV summaries, diagonal statistics, and route thresholds
    sol_streaming.py exact/approximate streamed online-softmax update

packages/seqattn-multigpu/
  planning.py       immutable per-device plans and static query partitioning
  streaming.py      static and feedback-driven multi-device execution
  dynamic.py        measured dynamic Q controller and query cursor
  projection.py     multi-device materialized QKV producer
  dit.py            specialized fused H3 consumer pipeline
  stats.py          plugin-owned execution statistics
```

`seqattn_core` exports the single-GPU API. It does not export `MultiGpu*`
symbols. The optional `seqattn-multigpu` distribution consumes the versioned
private plugin bridge and must match the core version exactly.

## Execution families

Backend selection and model-specific DiT settings share the same TOML document
but retain separate tables and dataclasses. Attention owns only backend policy;
each model owns its execution modes and projection/FFN tile topology. H3 and
Wan use the same `execution_mode`, `projection_tile_tokens`, and
`ffn_tile_tokens` names. LTX2 uses the same projection name but keeps separate
video/audio FFN tiles. There is no generic `[projection]` table, and attention
Q/KV chunks remain runtime inputs rather than deployment-wide model settings.

| Family | Input policy | Q/K/V policy | Output policy |
|---|---|---|---|
| Contiguous streaming | Caller-owned pinned CPU Q/K/V | Resident Q, streamed K/V | Pinned CPU output or device consumer |
| Projected attention | Caller-owned pinned hidden | Complete pinned host Q/K/V produced by callback | Output projection runs before D2H |
| Recomputed attention | Caller-owned pinned hidden | Q and K/V regenerated directly into CUDA tiles | Required device consumer |
| Paged attention | `PageSource` and `PageSink` | Bounded staging and optional bounded K/V cache | Page sink |
| H3 block runtime | Consumer callbacks and pinned hidden | Explicit materialized or recompute policy | Attention epilogue and MLP tiles |
| H3 Sol streaming | Same H3 callbacks | Projection-time materialized summaries or a BF16 summary pass, then exact/centroid-routed streaming | Required device consumer |

Choose a family from data ownership and capacity constraints. Paged execution
is for bounded host memory or page-backed storage; it is not a faster wrapper
around complete pinned Q/K/V tensors.

## Exact recurrence

For each packed segment, SeqAttn keeps one query super-block resident and scans
every K/V tile in that segment. Partial outputs are combined with FP32 online
softmax state:

```text
m_new = max(m_old, m_tile)
l_new = exp(m_old - m_new) * l_old + exp(m_tile - m_new) * l_tile
o_new = exp(m_old - m_new) * o_old + exp(m_tile - m_new) * o_tile
output = o_new / l_new
```

Running max, normalizer, and accumulator remain FP32 for BF16 and FP16 input.
Packed `cu_seqlens` are hard boundaries: no Q or K/V tile may cross a segment.
Exact dense attention observes all K/V for a segment before finalizing Q.
Causal alignment uses bottom-right positions when Q and K lengths differ.

## Approximate Sol recurrence

`sol_streaming` preserves the same resident-Q bound and FP32 online-softmax
state but changes the contribution of selected 64-token K/V blocks:

1. Each segment obtains BF16 K centroids and BF16 V sums, then computes FP32
   diagonal K-centroid statistics. Materialized H3 creates the summaries while
   projected K/V is still resident; BF16 sources use a separate prepass.
2. Each resident 64-token Q block computes a route threshold from its centroid.
3. The local +/-1 block band, configured exact Q/KV prefixes, and blocks above
   the threshold use exact token attention.
4. Other blocks use one centroid score with the V sum and account for the true
   token count in the online-softmax normalizer.

Packed segment boundaries remain absolute. Exact prefixes apply to both query
and key/value blocks and round outward to a full 64-token block. The path is
approximate because unrouted tokens share a block score and averaged value.

This changes arithmetic, not data availability: every Q chunk still consumes
all K/V blocks. Materialized H3 quantizes K/V with one symmetric FP16 scale per
64-token/head block during projection and transports INT8 K/V during attention;
its precomputed summaries avoid a raw BF16 K/V scan. Exact INT8 updates factor
the shared K scale through the QK product and the shared V scale into the
probability tile before PV, avoiding full elementwise K/V dequantization.
Recompute H3 performs one extra complete BF16 K/V projection pass for summaries,
then continues to regenerate BF16 K/V for each Q chunk. Standalone host-QKV uses
the BF16 summary prepass. Missing metadata, unsupported shape/dtype/device
contracts, causal mode, and insufficient workspace fail explicitly.

Sol query scheduling treats `q_chunk_tokens` as a maximum resident-Q capacity.
For multi-chunk segments it balances complete 64-token route blocks across the
same number of chunks, avoiding a short final Q tile that cannot cover K/V H2D.

## Workspace ownership

`workspace_budget_bytes` covers only CUDA allocations owned by SeqAttn:

- resident Q;
- FP32 max, normalizer, and accumulator state;
- one to three K/V staging slots;
- zero to two output slots, depending on output mode;
- a fixed allocator and launch margin.

It excludes the CUDA context, model weights, caller tensors, callback
temporaries, and the whole-process memory peak. The plan builder aligns manually
specified tiles to the selected kernel block dimensions.

The Sol plan adds K-centroid and V-sum arrays for the maximum segment block
count, FP32 per-head K statistics, per-resident-Q thresholds, and two route
counters. Dense and sparse execution borrow one `CudaWorkspace` and one
single-flight lock; their capacities are budgeted together.

Chunk axes are independent:

| Setting | Role |
|---|---|
| `q_chunk_tokens` | Resident Q super-block and FP32 state bound |
| `kv_chunk_tokens` | Streamed K/V transfer and update tile |
| `projection_tile_tokens` | Materialized hidden-to-QKV producer tile |
| `ffn_tile_tokens` | H3/Wan post-attention FFN tile |

Use [`q_chunk_calibration.md`](q_chunk_calibration.md) for deployment-specific
attention calibration. Do not infer projection or MLP tiles from the attention
workspace budget.

## Projection policies

Materialized projection has a global K/V readiness barrier:

```text
pinned hidden -> chunked projection -> complete pinned host Q/K/V
              -> exact streaming attention -> CUDA output projection
              -> pinned host output
```

The raw attention tensor is not round-tripped through CPU memory, but complete
host Q/K/V remains available for every resident Q pass.

The readiness barrier is required for exact global self-attention: an early
query cannot be finalized until projection has produced every key/value token
in its packed segment. Projection H2D, projection compute, and Q/K/V D2H remain
pipelined across tiles before that barrier. Compatible projected runners may
share a `MaterializedQKVArena`; allocation statistics report the persistent
arena capacity rather than only the current tensor views. Wan shares one arena
across self/text attention. LTX2 shares one video-query arena and one
audio-query arena across its six materialized attention stages.

Recomputed attention removes complete host Q/K/V:

```text
pinned hidden tile -> Q-only or K/V-only direct-write callback
                   -> planned CUDA tile -> shared attention executor
                   -> device output consumer
```

The tradeoff is repeated K/V projection for every resident Q pass. Hidden
staging is bounded by
`max(q_chunk_tokens, kv_chunk_tokens) * hidden_features`.

See [`design_dit_runtime.md`](design_dit_runtime.md) for H3 callback and buffer
contracts.

## Paged and storage path

```text
PageSource -> reader -> bounded DRAM K/V cache -> pinned staging
           -> CUDA attention workspace -> PageSink
```

The host budget covers operator-owned staging, direct-I/O bounce buffers,
metadata margin, and cache capacity. Caller tensors behind a
`MemoryPageSource` are outside that budget.

Persistent stores use a manifest plus aligned data files. `direct_io=True` is
a requirement, not a hint: unsupported filesystems, invalid alignment, short
I/O, or rejected `O_DIRECT` operations fail explicitly. See
[`paged_nvme_runtime.md`](paged_nvme_runtime.md).

`ProjectedCrossAttentionRunner` separates query projection from context K/V
projection. It supports unequal packed Q/KV lengths and GQA without accepting
arbitrary additive QxK masks. `RecomputedCrossAttentionRunner` writes Q and K/V
directly into independent device staging workspaces.

`RecomputedAttentionRunner` is independent of the materialized projection
pipeline. It accepts Q-only and KV-only callbacks that write one complete
attention Q or K/V tile directly into the CUDA attention workspace. It owns one
device hidden staging buffer sized to the larger attention tile and never
allocates host Q/K/V. Both paths use the same internal tile-source executor and
therefore the same online-softmax, finalize, and output-consumer schedule.

DiT block order is owned by architecture subpackages, while attention,
projection, output consumers, tiled host stages, and structured mask conversion
remain shared mechanisms. MiniMax-H3 keeps one-hidden materialized and
two-hidden recompute contracts. Wan runs self-attention, text cross-attention,
and FFN in that order. LTX2 materialized mode runs separate video/audio self
and text attention, materializes both audio/video cross directions before
either stream is updated, then runs separate FFNs. LTX2 recompute mode uses two
hidden buffers per modality: self-attention writes alternate buffers, text
cross-attention writes the original buffers, both bidirectional cross stages
read those same post-text snapshots and write the alternate buffers, and the
FFNs finish in place. Wan and LTX2 remain single-GPU integrations.

## Lifetime and concurrency

Runners own persistent CUDA buffers, streams, and events and are single-flight.
Reuse one runner for compatible serial calls; create separate runners for true
concurrency.

Callbacks must enqueue all work on the current runner stream before returning.
Returned or direct-written tensors must remain valid until that stream has
consumed them. Failure recovery restores runner slot and event state, but does
not make consumer callbacks transactional.

Dense and Sol H3 calls on one constructed runner are serialized by the same
single-flight lock because they reuse the same CUDA buffers and events.

## Correctness boundaries

- Execution is inference-only.
- Dense BF16/FP16 attention remains exact; `sol_streaming` is explicitly
  approximate.
- BF16 and FP16 storage are exact with respect to the same online-softmax
  algorithm; INT8 K/V is explicitly approximate.
- Asynchronous host paths require pinned tensors unless an API explicitly
  disables that requirement.
- Device-output transforms, projected attention, recomputed attention, and H3
  runners require the built-in Triton backend.
- Physical NVMe claims require a measured device. Simulated NVMe validates
  scheduler behavior only.
