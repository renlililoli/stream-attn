# Architecture notes

## Package boundaries

The repository ships one Python package: `seqattn_core`. It owns both the
public API and all implementation modules. The former `seqattn` compatibility
facade is intentionally not part of the core-only distribution.

```text
seqattn_core/        public API and implementation; no compat facades
  api.py             functional public API
  config.py          execution policy dataclasses
  planner.py         workspace and budget planning
  stats.py           statistics dataclasses
  reference.py       FP32 online-softmax CPU reference
  validation.py      host tensor and sequence validation
  quantization.py    per-token-group INT8 quantization
  streaming/         contiguous CPU-DRAM -> HBM execution
    backend.py       config loading, SM policy, and capability checks
    flash_backends.py explicit FA2/FA3/FA4 partial-forward adapters
    flash_split_executor.py shared FA streaming and FP32 combine schedule
    workspace.py     persistent CUDA buffers, streams, and events
    executor.py      built-in Triton copy/compute/output schedule
    runner.py        validation, reference dispatch, and public runner
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
    types.py         projection callback contracts
    workspace.py     persistent projection streams and buffers
    runner.py        projected attention orchestration
    api.py           functional convenience API
  benchmarking/      benchmark CLIs and reusable instrumentation
    common.py        JSON, sequence bounds, RSS, and NVML sampling
    streaming.py     DRAM-backed and full-GPU benchmark
    paged.py         memory, simulated-NVMe, and NVMe benchmark
    projection.py    projected pipeline benchmark
  kernels/           Triton kernels and launch helpers
```

Public imports use `seqattn_core` directly. Installed commands retain the
`seqattn-bench*` names, but their entry points resolve to
`seqattn_core.benchmarking`. Compatibility-only module paths are not carried
forward in the minimum core release.

## Memory hierarchy

The paged operator spans five storage levels:

```text
NVMe aligned Q/KV records
    -> bounded ordinary-DRAM K/V cache
    -> pinned Q/KV/output staging rings
    -> resident-Q and streamed-KV HBM workspace
    -> Triton SRAM/register tiles
```

The GPU operator has two tiling levels:

- `q_chunk_tokens` is an HBM-resident query super-block.  It controls the FP32
  online-softmax accumulator and therefore most of the workspace footprint.
- `kv_chunk_tokens` is a streamed K/V tile.  Two buffers permit copy/compute
  overlap; a third buffer is available for systems where the copy engine needs
  more queue depth.

Within a K/V tile, the built-in Triton kernel uses `BLOCK_M x BLOCK_N`
attention tiles and never writes the score matrix to global memory. Optional
FA2/FA3/FA4 adapters replace only this partial-forward operation; they retain
the same host streaming schedule and merge normalized partial output with FP32
LSE on the GPU.

## Fused recurrence

Each K/V update maintains, per query/head, FP32 state:

```text
m = running row maximum
l = running softmax normalizer
a = running unnormalized output
```

For a new tile with scores `S`, the kernel computes the merged maximum,
rescales the old state, accumulates `exp(S - m) @ V`, and stores the updated
state in one launch.  Final normalization and cast are a separate fused kernel
executed once per query super-block.

## Pipeline invariants

- H2D never overwrites a K/V slot until its compute-free event fires.
- Q is not replaced until every K/V update using it has completed.
- A GPU output slot is not reused until its D2H-free event fires.
- Packed sequence boundaries are scheduler boundaries; no query super-block or
  K/V scan crosses a segment.
- Causal positions use bottom-right alignment for unequal Q/K lengths.

## Host memory contract

`HostMemoryPlan` is the allocation authority for a paged run. The default 8GiB
policy reserves category limits of 1GiB pinned staging, 512MiB direct-I/O
bounce buffers, and 128MiB fixed metadata. Cache capacity is planned from the
remaining bytes. All staging rings, bounce rings, cache slots, and output rings
are created before the page loop; an allocation that would cross a category or
total limit fails immediately.

Caller-owned tensors adapted by `MemoryPageSource`/`MemoryPageSink` are outside
this accounting. They preserve the original API but do not provide a fixed
whole-process RAM guarantee.

The K/V cache is split into a deterministic low-page-id hot region and a
rolling region. At the default 80/20 split, the hot pages remain resident across
query passes while pages outside the cache continue as a sequential scan. Q is
single-use and bypasses the long-lived cache. Output pages enter their sink as
soon as the output staging slot is ready.

## Direct-I/O store

`NvmeQKVStore` uses one file for Q and one for paired K/V records. Pages do not
cross packed segments. Each descriptor records global and segment-local token
offsets, valid/padded token counts, payload size, padding, and aligned file
location. K/V data for a page is contiguous so one `preadv` fills both staging
tensors. INT8 records append K/V scale arrays to the same record.

The runtime uses a thread-safe pool of anonymous mmap buffers. mmap base
addresses, file offsets, and I/O lengths satisfy the 4096-byte `O_DIRECT`
contract. A short operation is an error. Direct-I/O open/read/write failures are
reported; the runtime never switches to buffered I/O without an explicit
`direct_io=False` test configuration.

Writers fsync temporary data files, validate their final sizes, rename them to
their stable names, fsync the directory, and publish `manifest.json` last.
Cancellation or an exception removes unpublished temporary/final data.

## CPU and GPU pipeline

The CPU thread pool only performs page I/O, cache copies, page packing, output
writes, and optional one-time quantization. It does not compute QK or PV. K/V
future reads are submitted up to the fixed queue depth. A pinned staging slot
is not reused until its H2D event completes; an HBM K/V slot is not reused until
its compute-free event completes; an output host slot is not reused until its
sink future completes.

The paged statistics include wall time, I/O time, queue wait, cache lookup,
copy traffic, compute-stream timing, output writes, cache hit ratio, and host
allocation peaks. End-to-end wall time includes visible I/O stalls.

## INT8 K/V

INT8 is an explicitly approximate storage mode. Quantization is symmetric per
64 tokens and KV head, with FP16 scales. The Triton update kernel loads INT8 K/V
and scales together and applies dequantization in the QK/PV path. No complete
BF16 K/V tensor is produced. CPU reference execution dequantizes only the
current page for auditability.

## Projected inference pipeline

`ProjectedAttentionRunner` adds model-projection producer/consumer hooks around
the Triton attention runtime:

```text
CPU hidden
    -> double-buffered hidden H2D
    -> GPU QKV projection callback
    -> Q/K/V D2H into persistent pinned backing buffers
    -> global K/V readiness barrier
    -> Q-resident / KV-streamed Triton attention
    -> GPU output-projection callback
    -> projected output D2H
```

The readiness barrier is required for exact global self-attention: an early
query cannot be finalized until projection has produced every key/value token
in its packed segment.  Projection H2D, projection compute, and Q/K/V D2H are
still pipelined across chunks before that barrier.

The important fusion is on the consumer side.  Raw attention output remains on
GPU and is passed directly to output projection.  Compared with a staged
implementation, this removes exactly one raw-attention D2H and one matching H2D
for every query token.  The callback may also include inference-only epilogues
such as gate/residual application.

An opt-in `output_mode="device_consumer"` finalizes into the Q buffer and removes
the separate raw-output HBM allocation. Q is not reusable until the consumer
has finished reading it; consumer results use `record_stream()` for D2H
lifetime. This mode is not the latency default because the August 18, 2026
61,312-token diagnostic measured 850.8ms with Q reuse and 828.7ms with the
same-run separate-output GPU-consumer path.

QKV and output projection are callbacks rather than hard-coded matmul kernels.
This permits ordinary BF16/FP16 linear layers, quantized modules, model-specific
Q/K normalization, and rotary embedding while the attention core remains
Triton.  The caller is responsible for keeping projection weights resident for
the duration of each phase.

## Planned follow-ups

- Shape-specific autotuning cache for block sizes, warps, stages, and K/V ring
  depth.
- io_uring and GPUDirect Storage implementations under the existing
  `PageSource`/`PageSink` contract.
- Validate and tune the existing FA3 and FA4 adapters on SM90 and Blackwell,
  including their lower-level preallocated-output interfaces.
- Optional prefetched residual/epilogue buffers so model-specific residual H2D
  overlaps the final K/V scan.
- Projection callback variants that can write into persistent GPU output slots
  and avoid temporary allocator traffic for standard dense linear layers.
- Backward kernels only after the inference API and memory contract stabilize.
