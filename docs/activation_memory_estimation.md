# Full H3 block estimation and allocation timelines

`seqattn_core.estimation.h3` models the complete **dense, inference-only H3
single-flight block** for materialized and recompute execution. It follows the
core runner, projection producer and `H3DeviceOutputConsumer`, with explicit
callback implementations. The previous per-Q generic FFN template has been
removed from the unreleased estimation API.

The estimator is offline and does not load a checkpoint or import a framework
adapter. Device pools, compute engines and transfer resources are named data,
so the same schedule can describe GPU/NPU/TPU deployments. Kernel costs and
private workspaces remain implementation-specific profiles. This adds no NPU,
TPU or other execution backend to SeqAttn.

## What is modeled

The selected path includes:

- materialized global projection tiles, separate hidden H2D and Q/K/V D2H,
  strided-result packing, keepalive slots, and a complete K/V readiness barrier;
- recompute Q-only/KV-only callbacks at the actual attention tile sizes,
  normalized hidden staging, returned temporary tensors and direct-write copies;
- one resident Q/output allocation, K/V slots, FP32 online-softmax state,
  per-segment Q/KV ranges and attention finalization;
- output projection, residual H2D on the compute stream and in-place residual
  updates;
- FFN carry across Q chunks, full and partial carry copies, direct views when a
  whole FFN tile fits, final tail flush, and output-slot reuse after D2H;
- norm2, modulation where enabled, FC1, SwiGLU, FC2, FFN residual and final D2H;
- optional norm1/modulation, position conversion/H2D, RoPE preparation and
  materialized in-place or recompute out-of-place Q/K normalization/RoPE;
- declared resident or physically stage-evicted weight groups and optional
  implementation workspaces in device memory or additional local pools.

The `block25` callback variant matches the retained benchmark: norm1,
modulation and QK norm/RoPE are omitted, while norm2 and the residual/MLP remain.
The `modulated` variant describes the ordinary no-LoRA production callbacks.
Full QKV, Q, KV, output projection, FC1 and FC2 have separate operator profiles;
there is no requirement to assume one GEMM throughput for all shapes.

Dense attention observes complete K/V per segment. Projection and pointwise
FFN ranges follow the actual global producer/consumer ranges and may straddle
attention-segment boundaries; attention tiles themselves never do. FFN calls
are `ceil(total_tokens / ffn_tile_tokens)`, not a sum of per-Q rounding.

## Live web parameter editor

Start the local web UI from an installed development environment:

```bash
PYTHONPATH=src python -m seqattn_core.estimation.web --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` in a browser. No frontend build, Node installation,
network assets or additional Python web framework is required. The browser calls
the same Python H3 estimator; it does not implement a second simulation model.
The service itself does not run accelerator kernels.

The form covers model shape/dtype, packed segments, Q/KV/projection/FFN tiles,
materialized/recompute comparison, attention/GEMM/vector throughput, copy
bandwidth, buffer counts, weight ownership and storage budgets. Optional
per-operator GEMM rates override the common GEMM value. Set **对比 FFN tiles**
to a comma-separated list to compare candidates and select the smallest eligible
activation footprint at the requested throughput target.

Edits automatically compute after a 400 ms debounce. Only the newest input can
replace the chart; stale or invalid results are visibly marked. Automatic update
can be disabled for manual batches. The embedded report retains linked cursors,
time zoom, storage/lifetime filters and JSON/SVG export. **下载 HTML** exports a
standalone report for the completed parameter set. Parameter JSON can be imported
or exported, and the last successful settings are stored in that browser.

A full `H3DeviceProfile.to_dict()` JSON can also be imported. Imported profiles
control operator rates/workspaces and disable the manual throughput controls;
filled UI capacity fields override the corresponding imported pool limits.
Shape-bound samples still validate against the model inputs. Clearing the
profile returns to manual rates. Initial manual rates are explicitly examples,
not measurements of a named accelerator.

The server binds to loopback by default. When running on a remote machine, use
an SSH tunnel or your editor's port forwarding, for example:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

The UI serializes requests and the service permits one active simulation at a
time. Interactive requests have a 2 MiB body limit, at most 16 candidates and a
conservative 120,000 event/scaling-work bound, checked before trace construction.
Oversized combinations report an actionable error; the unrestricted Python API
remains available for larger studies. The server exposes only its page, schema,
health and simulation endpoints; it does not serve repository files or accept
paths to load profiles from the server's filesystem.

## Generate a complete H3 report

```bash
PYTHONPATH=src python benchmarks/activation_timeline.py \
  --callback-variant modulated \
  --linear-memory int8_eager \
  --weight-scale channel \
  --output /tmp/h3-activation-timeline.html
```

This example uses H3 dimensions (hidden 5376, FFN 14336, 56 heads × 128),
81,159 tokens, and explicitly **synthetic** performance rates. It compares both
execution modes and FFN tiles of 2048, 4096, 8192 and 16384. Replace the rates
with matching calibration, or pass `--profile-json` to load a full operator
profile. It writes offline HTML and canonical JSON without a server or CDN.

The HTML has compute/I/O lanes, a selectable memory backdrop, per-pool stacked
component occupancy, active-vs-allocated curves, physical buffer lifetimes,
linked cursors, time zoom, candidate selection, and JSON/SVG export. Zero-time
synchronization milestones remain in JSON but are not drawn as compute work.

## Python API

```python
from seqattn_core.estimation import (
    H3BlockShape, H3CallbackConfig, H3DeviceProfile, H3ExecutionConfig,
    MemoryPool, RateProfile, build_h3_block_execution,
    estimate_activation_memory, write_timeline_report,
)

shape = H3BlockShape(
    segments=(81159,), hidden_features=5376, ffn_features=14336,
    heads=56, head_dim=128, activation_dtype="bfloat16", element_bytes=2,
)
# Illustrative values only. Use effective measured rates, not advertised peaks.
profile = H3DeviceProfile.from_rates(
    "illustrative profile",
    device_pool=MemoryPool("accelerator.memory", allocation_alignment_bytes=512),
    host_pool=MemoryPool("host.DRAM"),
    attention=RateProfile("attention FLOP/s", 200e12, ("compute",)),
    gemm=RateProfile("GEMM FLOP/s", 120e12, ("compute",)),
    vector=RateProfile("vector elements/s", 100e9, ("compute",)),
    h2d=RateProfile("H2D byte/s", 50e9, ("H2D",), kind="io"),
    d2h=RateProfile("D2H byte/s", 40e9, ("D2H",), kind="io"),
    d2d=RateProfile("D2D byte/s", 500e9, ("compute",)),
)
callbacks = H3CallbackConfig(
    variant="modulated", linear_memory="int8_eager",
    per_channel_weight_scale=True,
)
specs = [
    build_h3_block_execution(
        shape, H3ExecutionConfig(3840, 4096, 4096, ffn), profile,
        callbacks=callbacks,
    )
    for ffn in (2048, 4096, 8192, 16384)
]
result = estimate_activation_memory(
    specs, objective_pool="accelerator.memory",
    objective_owners=frozenset({"operator", "callback", "caller"}),
    target_throughput_fraction=0.95,
)
write_timeline_report(result, "h3.html", json_path="h3.json")
```

`H3ExecutionConfig.from_attention_plan(plan, projection_tile_tokens=...,
ffn_tile_tokens=...)` preserves resolved attention capacities and host-arena
capacity from an existing public plan. The standalone builder accepts resolved
capacities; it does not guess a device's legal tiles. `q_alignment` and
`kv_alignment` in the device profile enforce declared constraints.

Materialized execution aliases destination hidden to source hidden. Recompute
has distinct source/destination host storage. Host Q/K/V arenas are allocated
only for materialized execution. Their capacities can exceed the actual input
for reusable runners. The carry and final-output slots use the full configured
FFN capacity even when the sequence has a shorter tail.

## Operator timing and private memory

`H3OperatorProfile` combines an explicit `RateProfile`, exact-shape
`H3OperatorSample` records, or both. Samples take precedence. Missing sample
shapes fail unless a rate model was explicitly provided; they are never silently
interpolated. Sampled profiles must be bound with `profile.for_shape(shape)` so
heads, widths, dtype, RoPE and conditioning layout cannot accidentally change.
Sequence length itself is not part of that binding, allowing matching tiles to
be reused across sequence lengths.

Work units are explicit by operator family:

| Keys | Rate units |
|---|---|
| `attention`, `qkv`, `q`, `kv`, `out`, `fc1`, `fc2`, `adaln`, `swiglu_fc2` | FLOP/s; multiply-add counts as two FLOPs |
| `h2d`, `d2h`, `d2d` | Payload byte/s |
| Norm, modulation, RoPE, finalize, SwiGLU and residual operators | Processed element/s |

Transfer samples require `work=payload_bytes`: the same token count may describe
hidden, Q/K/V, position, or weight payloads with different widths. Attention
samples also include `kv_tokens`. A profile measures exactly the named operator,
not its entire surrounding callback. For example, `fc1` excludes norm2 and
`qkv` excludes norm1/QK-RoPE, which are separate graph nodes.

Additional workspace excludes the explicitly modeled input/output tensors.
There are three selectable linear-memory contracts:

- `dense`: visible tensor intermediates, plus any declared operator workspace;
- `int8_eager`: source-derived eager ConvRot/I8 accounting: rotated input,
  in-place input-dtype quantization, row padding, INT32 accumulator, transposed
  weight temporary, chunked FP32 scaling, retained converted parts, and final
  concatenation. Tensorwise/per-channel weight scale and ConvRot group are
  explicit settings. This is not a default for fused kernels or other backends;
- `profile`: implementation-private memory comes exclusively from supplied
  operator samples or explicit workspace models.

For sampled operations, measured private workspace replaces the analytical
internal estimate, preventing double counting. `fused_swiglu_fc2=True` requires
its own `swiglu_fc2` profile including fused activation workspace. Recompute
callbacks returning fresh tensors and copying to runtime destinations are the
default. A genuinely direct-writing callback can be declared explicitly; it
cannot be combined with the eager INT8 implementation that returns a tensor.

`H3ScratchBuffer` and `H3DeviceProfile.local_pools` describe extra operator
workspaces in, for example, `NPU.core0.UB` or TPU local memory. Pool names represent
physical capacities and must be unique; aliases share a pool. Capacity checks
cover all pools even when optimizing only the main device activation pool.

## Measure the actual operator instead of guessing its workspace

```python
from dataclasses import replace
from seqattn_core.estimation import H3OperatorProfile, measure_cuda_h3_operator

# x and the operator's weights already exist; this closure runs one FC1.
measurement = measure_cuda_h3_operator(
    lambda: fc1(x),
    tokens=x.shape[0],
    output_allocation_bytes=x.shape[0] * (2 * ffn_features) * x.element_size(),
    warmup=2, repeats=5,
    provenance="record GPU, backend/compiler, dtype, tile and weight layout here",
)
profile = replace(profile, operators={
    **profile.operators,
    "fc1": H3OperatorProfile(
        samples=(measurement.sample,), provenance=measurement.provenance,
    ),
}).for_shape(shape)
```

Supply one sample for every used full/tail shape, or explicitly retain a rate
fallback. New output storage is zero for in-place/direct-write output. Shared
views count their physical backing once. The helper uses median CUDA-event time
and maximum observed extra workspace, preserving raw repeats and device/version
metadata. It synchronizes CUDA and resets allocator peak counters; use an idle
process/device. Any operation retaining new allocations after its result is
released fails until that storage is declared persistent. Warmup is separated from measurements. Event intervals may include idle gaps
between submissions inside an operator; CPU work outside the event interval
and allocator cached pages are not measured.

This CUDA helper is optional; NPU/TPU tools can populate the same sample schema.
`profile.to_dict()` and `H3DeviceProfile.from_dict()` support JSON round trips.
Freeze the implementation version, CPU/NUMA placement, dtype, physical layout,
weight layout/residency and tile shape with calibration provenance.

## Memory scopes and search

The graph labels owners explicitly:

| Owner | Examples |
|---|---|
| `operator` | Resident attention state, carry, output slots, staging, host QKV arena, copy packing |
| `callback` | Norm/linear/activation outputs, position casts, declared private operator workspace |
| `caller` | Input/output hidden, supplied conditioning and position metadata |
| `weights` | Explicitly declared resident or stage-evicted weight groups |

`H3WeightPolicy()` defaults to zero declared weights, matching an activation
estimate. Nonzero resident weights are counted without timing transfers.
`mode="staged"` adds H2D and requires the declared physical eviction at the
projection/consumer context boundaries. A lease retaining cached weights should
be modeled as resident, not assumed to free memory at context exit. Recompute
holds projection and consumer weights concurrently.

`memory_statistics(trace, owners=...)` restricts reported occupancy by owner.
Search accepts `objective_owners` to minimize activation peak while still
checking **all** declared allocations against every pool's capacity. The report
shows this objective separately from total physical pool occupancy.

Core persistent tensor bytes, the fixed workspace allowance and the resulting
core workspace budget are distinct metadata fields. The default 32 MiB margin
is not an allocation and does not appear in the physical memory curve. Context,
allocator reserved/cached pages and unlisted caller allocations are excluded.

`minimum_capacity` selects the smallest objective peak among capacity-feasible
candidates. `selected` also satisfies whole-block latency: by default, at least
95% of the fastest supplied candidate's predicted throughput. The reference is
computed before capacity filtering. `max_latency_seconds` adds an absolute
constraint. An unreachable target returns `None` and retains all candidates.
Minima are over the supplied candidates and fixed schedule, not every possible
tile or an arbitrary global scheduling optimum.

## Lifetimes and synchronization

Streams preserve submission order independently of physical resource names.
Zero-time control milestones represent waits, slot reuse, barriers, callback
returns and final synchronization. There is no fictitious computation time for
an event. Named `BufferSpec` allocations count aliased views once.

`allocate_before` / `release_after` anchors represent reference lifetimes that
extend beyond the last kernel use, such as projection keepalive and parent
post-attention tensors. Releases at a timestamp precede reuse at that timestamp.
Persistent allocations remain occupied at the end of the reporting window.
The active curve shows operation use; the allocated curve also includes idle
but retained buffers. Component maxima are not added together to invent a total
peak: the total peak is computed at a single actual modeled instant.

Predicted lifetimes use operation boundaries. They are not allocator traces;
CPU dispatch timing and asynchronous allocator caching can differ. Imported
`trace_from_measurements()` timestamps use the same visualization, with declared
buffer ownership kept explicit. GPU timestamps alone do not establish hidden
CPU reference-release times.

## Validation and limits

Tests compare FFN ranges/carry/tail flush directly with the real
`H3DeviceOutputConsumer`, and compare persistent bytes with runtime estimators.
Small real-CUDA tests run both H3 runners and compare Q/KV/projection/FFN ranges
and actual persistent tensor bytes. Additional CUDA tests compare the selected
eager I8 workspace model against the real local implementation at H3 FC1 shapes.

The full-block replay is described in
[`benchmark_h3_full_estimator_2026-09-08.md`](benchmark_h3_full_estimator_2026-09-08.md).
Its core counts/workspace match the retained block-25 results. Kernel timing
accuracy remains conditional on matching tile calibration; a full native GEMM
rate is not a small-tile saturation curve.

This model covers the selected dense materialized/recompute paths and no-LoRA
callback variants. Sol/sparse and multi-GPU task-consumer execution require
separate models and are not silently treated as dense carry execution. Checkpoint
loading, whole-model weight eviction and framework integration remain consumer
responsibilities. The deterministic `build_attention_plan()` API is unchanged.
