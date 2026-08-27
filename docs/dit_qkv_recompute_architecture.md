# DiT QKV recompute architecture

Date: 2026-08-27

## Scope

SeqAttn exposes two explicit MiniMax-H3 block schedulers. The caller selects
the storage policy; the core does not infer a mode from available host memory.

- `H3MaterializedRunner` projects small hidden tiles into sequence-sized pinned
  Q/K/V backing, then runs exact streaming attention.
- `H3RecomputeRunner` keeps two pinned hidden tensors and regenerates Q and K/V
  directly into the attention workspace at the attention tile sizes.

The core package has no ComfyUI or comfy-kitchen dependency. Model adapters
provide projection callbacks and weight leases.

## Public projection contracts

Materialized projection retains the existing complete-QKV callback:

```text
QKVProjector(hidden_tile, start, stop) -> (q, k, v)
```

Recompute uses separate direct-write callbacks:

```text
QTileProjector(hidden_tile, destination_q, start, stop) -> None
KVTileProjector(hidden_tile, destination_k, destination_v, start, stop) -> None
```

Each callback receives one complete attention tile. A Q callback range is
exactly one planned `q_chunk_tokens` range, including the final tail. A K/V
callback range is exactly one planned `kv_chunk_tokens` range. Materialized
`projection_chunk_tokens` never affects recompute callback ranges or counts.

The direct-write interface leaves model-specific fusion outside the core. A
MiniMax-H3 adapter can combine normalization, modulation, INT8 ConvRot row
selection, Q/K normalization, RoPE, and destination writes in one callback.

## Shared attention executor

The Triton executor consumes an internal `QKVTileSource` protocol. Both source
implementations enter the same online-softmax, finalize, and output-consumer
loop:

```text
HostQKVTileSource
    pinned Q/K/V -> H2D stream -> q/kv ready events

RecomputedQKVTileSource
    pinned hidden -> H2D stream -> compute-stream Q-only or KV-only callback

shared executor
    load Q
    scan K/V tiles
    update FP32 online-softmax state
    finalize
    invoke device output consumer
```

The host source preserves Q/K/V transfer events and K/V double buffering. The
recompute source does not need Q/K/V free events because projection and
attention consume those buffers on the same compute stream. It owns one hidden
staging allocation with shape:

```text
[max(q_chunk_tokens, kv_chunk_tokens), hidden_features]
```

The hidden staging free event is recorded after the projection callback has
enqueued its work. The H2D stream may then refill the staging allocation while
the compute stream consumes the projected Q or K/V tile.

## DiT runner split

The old combined `H3DiTRunner` API is removed. The public contracts are:

```text
H3MaterializedProjection(project_qkv, weight_lease)
H3RecomputeProjection(project_q, project_kv, weight_lease)
H3BlockOps(attention_epilogue, mlp, consumer_lease)
```

`attention_epilogue` always receives the residual host tensor explicitly:

```text
attention_epilogue(attention, residual_host, start, stop) -> device_hidden
```

`H3MaterializedRunner.run_block_` uses one pinned hidden allocation in place.
Projection has consumed the complete source before the consumer overwrites any
range.

`H3RecomputeRunner.run_block` requires distinct source and destination pinned
hidden storage. The source remains immutable for the complete block because
every later query range must regenerate K/V from the original block input.
`run_blocks_` swaps the two physical buffers after every block and returns the
buffer containing the final result.

## Plans and memory

`H3MaterializedPlan` contains:

```text
projection_chunk_tokens
q_chunk_tokens
kv_chunk_tokens
mlp_chunk_tokens
estimated_workspace_bytes
```

`H3RecomputePlan` contains:

```text
q_chunk_tokens
kv_chunk_tokens
mlp_chunk_tokens
hidden_staging_tokens
estimated_workspace_bytes
```

There is deliberately no projection chunk in the recompute plan. Ignoring
bounded device workspaces and metadata, logical host activation is:

```text
materialized = N * (H + 3A) * element_size
recompute    = N * (2H) * element_size
```

where `A = heads * head_dim`.

## Validation

CUDA tests cover full-block parity, packed sequences with empty segments,
different Q/KV chunks, non-divisible tails, exact projector call ranges,
source immutability, two-block physical-buffer ping-pong, absence of host Q/K/V
on the recompute runner, and bounded allocator growth across repeated runs.

The standalone benchmark
[`benchmarks/minimax_h3_convrot_qkv_recompute.py`](../benchmarks/minimax_h3_convrot_qkv_recompute.py)
uses synthetic pinned BF16 hidden and the real MiniMax-H3 block 25 INT8 ConvRot
weights. It slices Q rows and the contiguous K/V rows together with their
per-output scales and optional bias. Before timing, it requires the concatenated
Q-only/KV-only result to match the complete QKV operator element by element.

The authoritative benchmark environment is the Community ComfyUI runtime, not
the DiffSynth development service. The benchmark defaults to the standard
Community image model layout under `/opt/ComfyUI/models`, loads the checkpoint
through `comfy.sd.load_diffusion_model`, and rejects environments that do not
match these pinned contracts:

```text
base image = comfyui@sha256:4708ab49a718640950f5cd698172d4800718d3b62e961f79d20866c115a8cff5
ComfyUI = 0.30.0, commit 9a9fdb10ed144ce760d9682cb247526ea23cc525
PyTorch = 2.10.0+cu128
PyTorch CUDA = 12.8
comfy-aimdo = 0.4.11
```

The benchmark initializes AIMDO before importing Torch or ComfyUI model code,
matching the Community custom-launcher requirement. A DiffSynth-derived image
may be useful for development, but results from it are not authoritative for
this Community ComfyUI A/B protocol.

The formal A/B defaults are:

```text
tokens=262720
q_chunk_tokens=16384
kv_chunk_tokens=4096
mlp_tile_tokens=2048
materialized qkv_tile_tokens=2048
warmup=1
repeats=3
```

`compare` launches materialized and recompute in separate child processes. It
records median/mean wall time, Torch allocated/reserved peaks, PID NVML peak,
RSS peak, and logical host activation bytes. Generated JSON is an experiment
artifact and is not committed.

Results from the removed projection-subtiled recompute implementation are not
comparable with this architecture. In particular, earlier `Q=196608` results
and any run where recompute ranges were controlled by
`projection_chunk_tokens` are invalid and must not enter conclusions.
