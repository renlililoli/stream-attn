# SeqAttn architecture

Status: current for `seqattn-core 0.3.0a4` on 2026-08-28.

SeqAttn is an inference-only exact dense-attention runtime for workloads whose
complete Q/K/V working set does not fit in GPU memory. The core package owns
generic planning, execution, storage, and callback orchestration. Model
loading, checkpoint conversion, block eviction, and UI integration remain
consumer responsibilities.

## Package boundary

```text
src/seqattn_core/
  api.py              functional contiguous-attention entry points
  config.py           immutable execution policies
  planner.py          workspace and tile planning
  streaming/          pinned-host Q/K/V streaming
  projection/         hidden-to-QKV and device-output pipelines
  paged/              fixed-host-budget page runtime
  storage/            aligned Q/K/V and output stores
  dit/                generic MiniMax-H3 callback schedulers
  kernels/            Triton kernels and launch profiles

packages/seqattn-multigpu/
  optional static and dynamic multi-GPU execution
```

`seqattn_core` exports the single-GPU API. It does not export `MultiGpu*`
symbols. The optional `seqattn-multigpu` distribution consumes the versioned
private plugin bridge and must match the core version exactly.

## Execution families

| Family | Input policy | Q/K/V policy | Output policy |
|---|---|---|---|
| Contiguous streaming | Caller-owned pinned CPU Q/K/V | Resident Q, streamed K/V | Pinned CPU output or device consumer |
| Projected attention | Caller-owned pinned hidden | Complete pinned host Q/K/V produced by callback | Output projection runs before D2H |
| Recomputed attention | Caller-owned pinned hidden | Q and K/V regenerated directly into CUDA tiles | Required device consumer |
| Paged attention | `PageSource` and `PageSink` | Bounded staging and optional bounded K/V cache | Page sink |
| H3 block runtime | Consumer callbacks and pinned hidden | Explicit materialized or recompute policy | Attention epilogue and MLP tiles |

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

## Workspace ownership

`workspace_budget_bytes` covers only CUDA allocations owned by SeqAttn:

- resident Q;
- FP32 max, normalizer, and accumulator state;
- one to three K/V staging slots;
- zero to two output slots, depending on output mode;
- a fixed allocator and launch margin.

It excludes the CUDA context, model weights, caller tensors, callback
temporaries, and the whole-process memory peak. The planner aligns manually
specified tiles to the selected kernel block dimensions.

Chunk axes are independent:

| Setting | Role |
|---|---|
| `q_chunk_tokens` | Resident Q super-block and FP32 state bound |
| `kv_chunk_tokens` | Streamed K/V transfer and update tile |
| `projection_chunk_tokens` | Materialized hidden-to-QKV producer tile |
| `mlp_chunk_tokens` | H3 post-attention MLP tile |

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

## Lifetime and concurrency

Runners own persistent CUDA buffers, streams, and events and are single-flight.
Reuse one runner for compatible serial calls; create separate runners for true
concurrency.

Callbacks must enqueue all work on the current runner stream before returning.
Returned or direct-written tensors must remain valid until that stream has
consumed them. Failure recovery restores runner slot and event state, but does
not make consumer callbacks transactional.

## Correctness boundaries

- Execution is inference-only.
- BF16 and FP16 storage are exact with respect to the same online-softmax
  algorithm; INT8 K/V is explicitly approximate.
- Asynchronous host paths require pinned tensors unless an API explicitly
  disables that requirement.
- Device-output transforms, projected attention, recomputed attention, and H3
  runners require the built-in Triton backend.
- Physical NVMe claims require a measured device. Simulated NVMe validates
  scheduler behavior only.
