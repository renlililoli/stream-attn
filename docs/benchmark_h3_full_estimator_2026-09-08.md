# Full H3 block estimator: implementation and validation

Date: 2026-09-08. The H3-specific model replaces the initial generic per-Q FFN
template. [API and calibration guide](activation_memory_estimation.md).

## Actual execution and ownership

The model follows `H3MaterializedRunner`, `H3RecomputeRunner`,
`MaterializedProjectionProducer`, the streaming executor/tile source and
`H3DeviceOutputConsumer`. The block-25 callbacks were checked against the retained
original benchmark script; the modulated callbacks were checked against the
consumer's production implementation without importing it into core.

Both execution modes include explicit stream submission order, projection/input
readiness, K/V reuse, FP32 state and finalize, residual H2D, FFN carry/emit/flush,
and output-slot D2H reuse. Materialized hidden is updated in place; recompute
uses separate host source and destination. Projection keepalive and parent
post-attention reference lifetimes are represented separately from kernel use.

The core workspace metadata is the sum of actual persistent CUDA tensor sizes
plus the separate 32 MiB allowance. This allowance is not drawn as a physical
allocation. Callback temporaries, declared weights and extra local-pool
workspaces have separate owners/components.

For the modulated callback path, the graph also includes modulation projection,
norm1/modulation, CPU position cast/H2D, RoPE construction, Q/K normalization and
rotation, norm2/modulation, and gated residual updates. The benchmark variant
explicitly omits norm1/modulation/RoPE, as the original benchmark did.

## Direct correctness checks

The focused suite passed **69 tests with CUDA enabled** on an idle RTX 5090 in
the existing `minimax-h3-seqattn:checks-release-0.4.4` image.

- Tests run the actual `H3DeviceOutputConsumer` state machine using CPU tensors
  and dummy events, comparing exact FFN ranges and carried row contents across
  Q chunks, packed segments and tails.
- Small real-CUDA tests execute both complete H3 runner modes with dense Torch
  callbacks. Q, KV, projection and FFN ranges match the model, and the sum of
  actual persistent CUDA tensor bytes equals the modeled core storage.
- CUDA calibration tests measure additional operator workspace separately from
  returned output storage.
- Eager INT8 tests call the real local implementation at H3 FC1 shapes: M=8192
  with tensorwise and channel scales, and M=16384 with channel scales. The
  analytical workspace agrees with measured CUDA allocator peak delta within
  2 MiB for these points. These are synthetic-weight operator tests, not a new
  checkpoint-level performance benchmark.
- Further tests cover exact-shape sample lookup, dtype/shape bindings, distinct
  transfer payloads at equal token counts, local memory pool capacity, weight
  ownership, physical aliases, and output-copy reuse constraints.

The INT8 model uses input-dtype division and in-place rounding/clamping, as the
selected eager implementation does. It tracks padded INT32 accumulators,
retained converted parts, overlapping Python loop locals, and the actual final
chunk size at concatenation. It does not assume three full FP32 input copies.
Other/fused implementations should supply measured workspace samples.

## Replay of the retained block-25 data

Source: parent-project
`workspace/benchmarks/results/seqattn_single_block25_81159_20260825/`.
Shape: 81,159 tokens, hidden 5376, FFN 14336, 56 heads × 128, BF16;
Q=3840 and KV=4096. The selected memory implementation is eager INT8 ConvRot
with per-channel weight scale, matching the rowwise ConvRot weight layout.

Unprofiled native CUDA-event stage means provide effective attention/QKV/output/
FFN rates. Copy bandwidth and a vector-rate proxy come from the accompanying
profile. **This is a rate-transfer diagnostic**, not independently measured
small-tile saturation curves. Primary comparison latencies are the unprofiled
wall-time JSONs; no streamed wall time or peak was fitted to the model.

All eight streamed configurations match the recorded FFN call counts and core
workspace bytes exactly. The seven configurations with raw repeat `max/min <=
1.1` have mean absolute time error **1.4374%**, maximum **3.1225%**. The variable
configuration-file run is retained and reported separately, not silently
removed: it ranges from 1.563 to 2.395 seconds and has a 2.254-second median.

| Projection tile | FFN tile | Predicted time | Measured median | Predicted activation peak | Observed Torch peak |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 1.487570 s | 1.535516 s | 1588.148 MiB | Not recorded |
| 4096 | 4096, variable run | 1.489487 s | 2.253628 s | 2112.062 MiB | 2090.736 MiB |
| 2048 | 4096 | 1.488513 s | 1.504778 s | 2039.438 MiB | 2048.736 MiB |
| 2048 | 8192 | 1.489387 s | 1.490434 s | 2975.047 MiB | 2984.346 MiB |
| 4096 | 8192 | 1.490362 s | 1.471253 s | 3017.047 MiB | 3026.346 MiB |
| 8192 | 8192 | 1.492311 s | 1.484030 s | 3101.047 MiB | 3110.346 MiB |
| 4096 | 12288 | 1.490362 s | 1.462356 s | 4016.312 MiB | 4025.611 MiB |
| 4096 | 16384 | 1.492110 s | 1.462620 s | 5138.364 MiB | 5147.664 MiB |

The activation estimate excludes undeclared weights and other persistent caller
allocations. The recorded Torch peak includes all live allocator tensors. Their
absolute values therefore have different scopes; no absolute activation-only
error percentage is claimed. Six regular memory-bearing cases have an
approximately 9.299 MiB residual, which is retained rather than fitted as an
arbitrary correction. The variable run also differs in projection-phase peak
and should not be used to infer a universal correction.

For fixed projection=4096, doubling FFN tile from 8192 to 16384 gives:

| Increment | Bytes |
|---|---:|
| New H3 predicted activation growth | 2,224,362,496 |
| Observed Torch peak growth | 2,224,363,008 |
| Difference | -512 |

This incremental comparison holds the block, weights and attention/projection
settings fixed. It demonstrates that the corrected carry and private INT8 tensor
lifetimes reproduce the large FFN memory growth; it does not prove an absolute
whole-process peak prediction on arbitrary devices or backends.

Canonical [comparison JSON](experiments/h3_full_block_estimator_20260908/comparison.json)
contains every raw wall-time sample, source fingerprints, calibration inputs,
per-case counts/bytes, limitations and the incremental calculation.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/validate_h3_block_estimation.py \
  --directory ../../workspace/benchmarks/results/seqattn_single_block25_81159_20260825 \
  --output-dir /tmp/seqattn-full-h3-estimation

PYTHONPATH=src python benchmarks/activation_timeline.py \
  --callback-variant modulated --linear-memory int8_eager --weight-scale channel \
  --output /tmp/h3-full-timeline.html
```

The first command replays historical observations and emits a benchmark-variant
HTML trace and JSON. The second generates a complete modulated H3 timeline with
explicitly synthetic default rates, or a supplied `--profile-json`. Neither
loads a checkpoint or executes model kernels. Use `measure_cuda_h3_operator`
with the consumer's actual operators to replace rate transfer with calibrated
full/tail shape samples before choosing a deployment default.
