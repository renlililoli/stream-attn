# Whole H3 block behavior audit — September 9, 2026

This audit checks **behavior independently of time** across the complete dense
H3 block. It covers both materialized and recompute runners with the real
production block-25 callbacks. It is not a claim that arbitrary backend kernels
or allocator internals are simulated exactly.

## Corrections

| Area | Previous mismatch | Corrected contract |
|---|---|---|
| Projection concurrency | Packing and GEMM always mutually exclusive | Profile can declare the packing concurrency observed on RTX 5090 |
| Host submission | Position H2D only ordered the GPU stream | Blocking `.to(device)` gates all subsequent host submissions; slot/global barriers also advance the host gate |
| Host position cast | CPU cast lifetime began at a GPU compute milestone | Cast is anchored on the host submission gate |
| Production MLP | Default model split SwiGLU from FC2 | Production defaults to `linear_input_act(..., "swiglu")`; its internal workspace belongs to the selected backend |
| One-token QKV tail | Unconditionally packed strided-layout results | Singleton views are contiguous and copied directly; no packing buffer or kernel |
| Recompute Q lifetime | Raw Q retained through destination copy | Raw Q ends when rebound to the RoPE result; raw KV remains through V copy |
| Recompute RoPE lifetime | RoPE table could end before concatenation | Retained through callback concatenation/return |
| Configuration | INT8 split MLP or contiguous production QKV could describe a different callback silently | Incompatible production declarations fail explicitly |

A combined `swiglu_fc2` operator means the actual **API entry point**, not a
promise of one fused GPU kernel. The eager implementation applies activation
before rotation, while another implementation may fuse it into quantization.
The eager workspace model now accounts for that specific input-act lifetime and
is checked against real CUDA allocations at full and single-token shapes.

## Full production recording

Each mode was run in an independent process on idle RTX 5090 GPU 3, using the
same real checkpoint and CPU/NUMA settings as the projection audit. Each process
ran two full-block warmups, then recorded one full block with Nsight and Python
wrappers that observe the actual callbacks and kernel arguments.

- Tokens: **8193**, packed segments **4097 + 4096**.
- Q tile: 3840; KV, projection and FFN tiles: 4096.
- The workload deliberately creates a one-token projection/KV/FFN tail and
  FFN carry crossing both Q and packed-segment boundaries.
- BF16 hidden 5376, FFN 14336, 56 heads × 128, RoPE 96, real INT8 ConvRot weights.
- Materialized: 3 projection calls, 4 Q chunks, 6 attention updates, 3 FFN calls.
- Recompute: 4 Q projections, 6 KV projections, 6 attention updates, 3 FFN calls.

The complete observed operator order and token shapes match the corrected graph:
**30 compared operator events** for materialized, **44** for recompute.
The audit checks 28 behavioral assertions in materialized and 17 in recompute:

- projection, Q, KV, epilogue and FFN ranges;
- packed local offsets and first-KV `initialize=True` for every Q;
- FFN carry/direct-source choice, output slot, cross-Q count and final flush;
- finalize output aliases resident Q; FFN returns its input storage;
- epilogue residual uses new device storage; final hidden alias differs by mode;
- QKV views share backing storage; only noncontiguous results require packing;
- actual host transfer byte counts and core workspace budget;
- complete outputs are finite.

NVTX-attributed CUDA runtime records independently confirm **3** blocking
projection callbacks in materialized and **10** in recompute, matching the graph.
The capture is for behavior diagnostics, not primary latency measurement.

## Actual runtime dependency oracle

`tests/test_estimation_h3_dependencies.py` executes the **real**
`MaterializedProjectionProducer`, `TritonExecutorMixin`, `HostQKVTileSource`,
`RecomputedQKVTileSource` and `H3DeviceOutputConsumer` state machines with recording
streams/events and CPU tensors. It compares the transitive happens-before
relations on common operations against the estimator, without using durations.

Fourteen configurations cover:

- projection slots 1/2/3, with and without the observed blocking RoPE callback;
- materialized/recompute with KV/output slots 1/2;
- Q smaller/equal/larger than FFN tiles;
- carry, direct FFN views, packed boundaries and finish-time flushing.

This catches both missing dependencies and extra serialization. Kernel arithmetic
is replaced only in this dependency test; separate real CUDA runs validate actual
callbacks, final Q aliasing, core allocations and eager workspace sizes.

## Scope and remaining timing uncertainty

The verified contract is **single-flight, dense, noncausal H3, no LoRA**, with the
specified production callbacks or explicitly declared benchmark callbacks.
Sol/sparse, dynamic task-consumer execution, multigpu, unknown callback allocation
policies and alternative fused implementations are not implicitly equivalent.
Their behavior requires a separate supported contract/profile.

Atomic operator internals use an explicitly selected implementation or a measured
workspace sample. GPU occupancy, launch gaps, allocator bookkeeping and contention
can still cause time/peak approximation at that granularity. A low latency error
is not used as proof of behavioral consistency. The independent 64K projection
timing still has -11.20% error; that result remains recorded, not fitted away.

Historical full-block error figures remain tied to their archived source revision.
This audit does not relabel the old 524K timing as validation of the new model.

## Evidence and reproduction

Recordings, driver source, comparison outputs and immutable source hashes are in
[`experiments/h3_full_behavior_20260909`](experiments/h3_full_behavior_20260909/manifest.json).
Raw Nsight reports and SQLite files remain under
`/tmp/seqattn-h3-behavior-20260909/`, with hashes in the manifest.

```bash
PYTHONPATH=src python benchmarks/audit_h3_runtime_behavior.py \
  docs/experiments/h3_full_behavior_20260909/materialized.json \
  --sqlite /tmp/seqattn-h3-behavior-20260909/materialized.sqlite \
  --output materialized.audit.json
```

The consumer printed an existing destructor warning after results were saved;
both audit containers exited 0. No core runtime or consumer implementation was
changed to make the observed behavior agree with the estimator.
