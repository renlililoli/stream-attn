# seqattn

`seqattn` is an inference-only paged attention runtime for sequences larger
than a fixed GPU or host-memory working set. Q/K/V may live in caller-owned CPU
DRAM or in an aligned NVMe store. Query super-blocks stay resident in HBM while
K/V pages move through a bounded DRAM cache, pinned staging ring, and GPU ring.
The runtime never materializes the score matrix or requires complete paged
Q/K/V/output tensors in host memory.

The project follows the same IO-aware principle that makes FlashAttention
effective inside the GPU memory hierarchy, but applies it one level higher:

```text
FlashAttention:   HBM backing -> SRAM/register working set
seqattn tensor:   CPU DRAM backing -> HBM working set
seqattn paged:    NVMe backing -> DRAM cache -> pinned staging -> HBM working set
```

This is not a replacement for FlashAttention when full Q/K/V fit in VRAM.
It targets exact long-context inference under a fixed GPU-memory budget and
reports the resulting PCIe and latency cost explicitly.

## MiniMax-H3 integration

`seqattn` is integrated into a MiniMax-H3 NF4 inference branch in
[`renlililoli/minimax-h3-seq-chunk-attn`](https://github.com/renlililoli/minimax-h3-seq-chunk-attn).
The integration streams the complete H3 attention path rather than wrapping an
isolated attention microbenchmark:

```text
chunked QKV projection → pinned CPU Q/K/V → resident-Q Triton attention
                       → GPU out projection + gate + residual → CPU hidden
```

<p align="center">
  <img src="docs/assets/minimax-h3-live-overview.svg" alt="MiniMax-H3 132K-token live benchmark" width="100%">
</p>

### Results at a glance

| Experiment | Scale | Result | Why it matters |
|---|---:|---:|---|
| H3 8GB capacity probe | 132,288 packed tokens | **5,968 MiB** process peak | A complete 50-block denoise forward succeeds under an 8GiB whole-process target. |
| Native-vs-seqattn soak | 132,288 packed tokens | **native OOM after 14 steps** | `seqattn` reached the same point below 8GiB and continues running. |
| Projection pipeline | 61,312 tokens | **7,108 → 3,848 MiB** | Keeping attention output on GPU cuts the measured peak by **45.9%**. |
| Projection pipeline latency | 61,312 tokens | **919.79 → 843.44 ms** | Fusion reduces latency by **8.3%** as well as memory. |
| H3 integration vs prior streamed path | 61,056 packed tokens | **81.5 GiB less PCIe traffic/step** | Removes the raw-attention D2H→H2D round trip across 50 H3 blocks. |

The completed 132K capacity probe used the full 50-block DiT for one denoise
step, not a five-layer proxy.  It measured 236.39 seconds, 5,968 MiB PID-level
NVML peak, and 48.4 GiB CPU RSS.  The 50-step video-generation soak shown in
the figure is still running as of August 18, 2026; its live numbers are
published as an explicitly preliminary snapshot, not as a completed result.
The final report will also include both VAE decoders and MP4 muxing.

The native path is faster while it runs: 140.07 seconds per denoise step versus
about 224.31 seconds for `seqattn`, a 1.60× slowdown.  It does not complete this
run, however.  Unrestricted native DiffSynth reaches a 30,876MiB PID-level NVML
peak and OOMs while starting step 15 with a 3.53GiB allocation request.  At the
same 14-step checkpoint, `seqattn` remains stable at a 7,164MiB within-step peak
and about 4,432MiB at each step boundary.  Host RAM and PCIe costs are reported
rather than hidden.

See the [MiniMax-H3 integration report](docs/minimax_h3_integration.md) for the
protocol, completed measurements, live-soak boundary, and reproducibility
details.

### What “native” keeps in GPU memory

The native comparison is not a model-without-offload strawman.  It uses the
same CPU/DRAM weight backing and DiffSynth VRAM management; no disk offload is
used.  During denoising, DiffSynth onloads only the DiT and offloads the text
encoder and both VAEs.  The key difference is activation placement:

| Residency during the DiT loop | Native full sequence | `seqattn` |
|---|---|---|
| Inactive model weights | CPU DRAM | CPU DRAM |
| DiT weights | CPU-backed; prepared/current layers can reside on GPU | same backing, with frozen post-step-1 residency |
| Packed hidden/residual | complete tensor on GPU | complete tensor in pinned CPU DRAM; chunks on GPU |
| Q/K/V | complete tensors on GPU | complete tensors in pinned CPU DRAM; resident Q + streamed K/V |
| Attention output | complete tensor on GPU | consumed tile-by-tile by GPU out projection |
| MLP `fc1`, gate/up, product | complete sequence tensors on GPU | CPU intermediate with chunked GPU compute |
| Text encoder / Video VAE / Audio VAE | offloaded while DiT runs | offloaded while DiT runs |

<p align="center">
  <img src="docs/assets/minimax-h3-native-residency.svg" alt="Native MiniMax-H3 memory residency" width="100%">
</p>

At 132,288 BF16 tokens, one complete hidden/residual tensor is approximately
1.325GiB, full QKV is 5.299GiB, and the MLP `fc1` output is 7.065GiB.  Splitting
`fc1` produces 3.532GiB gate and up views; the native OOM is the attempt to
allocate another 3.53GiB for `SiLU(gate) * up`.  These are activation sizes,
not checkpoint-weight sizes.

## Features

- CPU-backed contiguous and pinned Q/K/V.
- Dense and packed variable-length APIs.
- BF16 and FP16 Triton kernels; auditable FP32 PyTorch reference.
- Non-causal and bottom-right-aligned causal attention.
- MHA, GQA, and MQA (`q_heads % kv_heads == 0`).
- Budget-aware query super-block planning.
- Fused QK, masking, online softmax, PV, and cross-tile state update.
- H2D, compute, and D2H streams with K/V ring buffers and optional double-buffered output.
- Persistent runner/workspace to avoid allocator growth across repeated calls.
- A fixed-budget paged API with memory, callback, and NVMe sources/sinks.
- Linux `O_DIRECT` files with aligned page records and explicit failure instead
  of silent buffered-I/O fallback.
- A deterministic two-region K/V cache: 80% low-page-id hot set and 20% rolling
  read-ahead by default.
- Optional INT8 K/V storage with FP16 scales per 64 tokens/head. This mode is
  approximate and must be selected explicitly.
- JSON benchmark output, NVML process peaks, logical PCIe traffic, and NVTX ranges.

V1 is inference-only: backward and dropout are intentionally unsupported.

## Relationship to Stream-CQSA

[Stream-CQSA](https://github.com/yiming-b/Stream-CQSA) and `seqattn` solve the
same capacity problem with different task decompositions.  Both keep complete
Q/K/V outside HBM and compute exact dense attention with a bounded GPU working
set.  Stream-CQSA recursively constructs combinatorial self-attention
subsequences; `seqattn` uses regular two-dimensional query-super-block by
key/value-tile scheduling.

For the default Stream-CQSA construction, recursion depth `t` produces `7^t`
subsequences of approximate length `N * (3/7)^t`.  In its current CPU-backed,
path-by-path gather implementation, the aggregate Q/K/V token occurrence is
therefore approximately:

```text
7^t * N * (3/7)^t = 3^t * N
```

The local FlashAttention calls also cover an aggregate dense score area of:

```text
7^t * (N * (3/7)^t)^2 = (9/7)^t * N^2
```

Interactions excluded by the CQS group mask are mathematically removed, but
the current modified FlashAttention path applies that mask after the local QK
matrix multiplication.  Consequently, recursive decomposition can increase
both host/device traffic and launched dense tensor-core work.

If `seqattn` divides Q into `r` resident super-blocks, its logical traffic is:

```text
H2D = |Q| + r * (|K| + |V|)
D2H = |O|
```

Every Q token is transferred once.  Each K/V tile is reused by all query rows
in the current resident super-block, and only the final FP16/BF16 output is
returned to the host.  With enough workspace for all Q rows (`r = 1`), Q, K,
and V are each transferred once.

Another important difference is reduction placement.  The current CPU-backed
Stream-CQSA path reconstructs per-subsequence FP32 numerator/denominator values
from FlashAttention output and LSE, transfers them to the CPU, and scatter-adds
them into full-sequence FP32 accumulators.  `seqattn` instead keeps FP32
online-softmax `(max, normalizer, accumulator)` state in HBM only for the
resident Q super-block.  It never reconstructs an absolute `exp(LSE)` global
denominator and does not require a full FP32 CPU numerator tensor.

This gives `seqattn` several advantages for regular dense inference:

- deterministic Q reuse across all K/V tiles in a super-block;
- lower logical H2D and substantially lower D2H volume at deep decomposition;
- no combinatorial task-count or masked-MMA amplification;
- stable FlashAttention-style online-softmax merging on the GPU;
- no full-sequence FP32 numerator accumulator in host RAM;
- an existing H2D/compute/D2H pipeline with reusable K/V ring buffers;
- native packed varlen, causal, cross-attention, GQA, and MQA interfaces.

The scope is deliberately narrower.  Stream-CQSA includes research forward and
backward paths and arbitrary CQS group masks.  `seqattn` V1 targets dense exact
inference only; it does not currently implement backward, dropout, arbitrary
sparse masks, a dynamic GPU page table, cache replacement, or cross-call page
residency.  Its HBM usage is best described as a statically scheduled resident
working set, not a general-purpose paging cache.

The comparison above is an implementation and I/O-complexity comparison, not a
claim based on a same-machine Stream-CQSA benchmark.  Direct comparisons should
use identical shapes, dtype, GPU-memory limits, CPU NUMA placement, and measured
H2D/D2H traffic.

## Installation

```bash
pip install -e '.[cuda,benchmark,dev]'
```

Linux, CUDA, PyTorch 2.5+, and Triton 3.1+ are the initial supported platform.

## API

The functional API mirrors the important parts of FlashAttention's dense and
varlen call shapes, but inputs and outputs are CPU tensors:

```python
import torch
from seqattn import StreamingAttentionConfig, streaming_attn_varlen_func

q = torch.randn(61_312, 56, 128, dtype=torch.bfloat16, pin_memory=True)
k = torch.randn(61_312, 56, 128, dtype=torch.bfloat16, pin_memory=True)
v = torch.randn(61_312, 56, 128, dtype=torch.bfloat16, pin_memory=True)
cu = torch.tensor([0, 61_312], dtype=torch.int32)

out = streaming_attn_varlen_func(
    q,
    k,
    v,
    cu,
    cu,
    max_seqlen_q=61_312,
    max_seqlen_k=61_312,
    causal=False,
    config=StreamingAttentionConfig(
        workspace_budget_bytes=4 * 2**30,
        kv_chunk_tokens=4096,
        backend="triton",
    ),
)
```

For repeated transformer layers or denoise steps, build one plan and reuse one
runner.  This keeps buffers, streams, and CUDA events persistent:

```python
from seqattn import StreamingAttentionRunner, build_plan

config = StreamingAttentionConfig(
    workspace_budget_bytes=4 * 2**30,
    kv_chunk_tokens=4096,
)
plan = build_plan(
    q_heads=56,
    kv_heads=56,
    head_dim=128,
    dtype=torch.bfloat16,
    device="cuda",
    max_q_tokens=61_312,
    max_kv_tokens=61_312,
    config=config,
)
runner = StreamingAttentionRunner(plan, config)
out = torch.empty_like(q).pin_memory()
runner(q, k, v, cu, cu, out=out)
```

Pass a reusable pinned `out` buffer in latency-sensitive loops.  Omitting it is
convenient, but allocating hundreds of MiB of pinned host memory can dominate a
short attention call and introduce allocator jitter.

### Projection-attention-output pipeline

`ProjectedAttentionRunner` connects a chunked GPU QKV producer to the Triton
attention runtime and consumes each finalized attention tile with an output
projection before D2H:

```python
from seqattn import (
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)

attention_config = StreamingAttentionConfig(
    backend="triton",
    workspace_budget_bytes=2 * 2**30,
    kv_chunk_tokens=4096,
    num_output_buffers=2,
)
pipeline_config = ProjectionPipelineConfig(
    projection_chunk_tokens=2048,
    num_projection_buffers=2,
)
plan = build_plan(
    q_heads=56,
    kv_heads=56,
    head_dim=128,
    dtype=hidden_cpu.dtype,
    device="cuda",
    max_q_tokens=hidden_cpu.shape[0],
    max_kv_tokens=hidden_cpu.shape[0],
    config=attention_config,
)
runner = ProjectedAttentionRunner(plan, attention_config, pipeline_config)

def project_qkv(hidden_gpu, start, stop):
    qkv = qkv_proj(hidden_gpu).view(-1, 56, 3, 128)
    # Model-specific Q/K normalization and RoPE can be applied here.
    return (
        qkv[:, :, 0, :].contiguous(),
        qkv[:, :, 1, :].contiguous(),
        qkv[:, :, 2, :].contiguous(),
    )

def output_projector(attention_gpu, start, stop):
    # This callback may also perform an inference-only gate/residual epilogue.
    return out_proj(attention_gpu)

projected_cpu = runner(
    hidden_cpu,
    cu_seqlens,
    project_qkv=project_qkv,
    output_projector=output_projector,
    output_features=hidden_size,
)
```

Exact global self-attention still has one unavoidable barrier: every K/V token
in a packed segment must be projected before a query result can be finalized.
The projection phase pipelines hidden H2D, QKV compute, and Q/K/V D2H across
chunks.  After the barrier, attention output stays on GPU and flows directly
into output projection, removing the old raw-attention D2H followed by H2D.

The projection callbacks are intentionally model-owned.  They can wrap normal
BF16/FP16 linear layers, quantized/offloaded modules, Q/K normalization, rotary
embedding, and output gate/residual logic while the attention core remains a
Triton operator.

`workspace_budget_bytes` only covers buffers owned by `seqattn`.  Whole-process
budgets must also reserve memory for the CUDA context, weights, and caller-owned
activations.

`output_mode="device_consumer"` additionally finalizes attention in the Q HBM
buffer and removes the separate raw-output HBM allocation. It is opt-in: the
61,312-token diagnostic on August 18, 2026 measured 850.8ms with Q reuse versus
828.7ms for the same-run separate-output GPU-consumer path, so reuse was 2.7%
slower and did not meet the 10% speedup threshold for becoming the default. The
mode remains useful when a fixed Q chunk makes removal of the output allocation
more important than latency; an auto planner may instead spend the freed budget
on a larger resident Q chunk.

### Fixed-host-memory paged API

The paged API does not require complete CPU Q/K/V tensors. `PageSource` reads
one caller-selected page into a preallocated staging buffer, while `PageSink`
consumes an output page immediately:

```python
from seqattn import (
    NvmeOutputSink,
    NvmeQKVStore,
    PagedAttentionConfig,
    PagedAttentionRunner,
    StreamingAttentionConfig,
)

store = NvmeQKVStore("/local-nvme/request-17", direct_io=True)
config = PagedAttentionConfig(
    host_memory_budget_bytes=8 * 2**30,
    direct_io=True,
    kv_storage_dtype="bf16",
    attention=StreamingAttentionConfig(
        workspace_budget_bytes=2 * 2**30,
        backend="triton",
    ),
)
runner = PagedAttentionRunner(config, device="cuda")
runner.run(
    store,
    store,
    cu_seqlens_q,
    cu_seqlens_k,
    NvmeOutputSink("/local-nvme/request-17-output", direct_io=True),
)
```

`MemoryPageSource` and `MemoryPageSink` adapt caller-owned CPU tensors to the
same interface. Complete tensors owned by the caller are not charged to the
operator budget, so the old tensor API and a complete `MemoryPageSink` output
are compatibility modes, not low-RAM execution.

The default 8GiB host policy reserves at most 1GiB for pinned staging, 512MiB
for direct-I/O bounce buffers, and 128MiB for fixed metadata. The remainder is
the DRAM K/V cache. Every runtime allocation is registered with
`HostMemoryPlan`; category or total-budget overruns fail before execution.
Q pages bypass the long-lived cache, and output pages are handed to their sink
as soon as D2H completes.

### NVMe store construction

`NvmeQKVWriter.from_tensors(...)` is a convenience path. Large requests should
use `NvmeQKVWriter.write_pages(q_pages, kv_pages)`, whose iterators yield one
already-sized page at a time. The on-disk layout is:

```text
manifest.json
q.bin
kv.bin
```

K and V for one token page share a single aligned record and one read. Payload
offsets, record lengths, and bounce buffers are 4096-byte aligned. Data files
are written under temporary names, checked and fsynced, then the manifest is
published last. With `direct_io=True`, unsupported filesystems fail explicitly;
`direct_io=False` exists for tests and is never an implicit fallback.

`NvmeOutputSink` publishes an analogous `manifest.json` plus `out.bin` without
allocating a full CPU output tensor. `CallbackOutputSink` supports immediate
application-owned consumption. Persistent stores are caller-managed;
`ephemeral_nvme_directory()` provides explicit temporary lifecycle management.

### Exact and INT8 modes

`kv_storage_dtype="bf16"` is the default exact mode. FP16 exact stores use
`"fp16"`. `"int8"` is optional and approximate: Q remains BF16/FP16, K/V use
symmetric INT8 quantization, and FP16 scales are stored per 64 tokens and KV
head. The Triton load path applies scales before QK/PV without creating a full
BF16 K/V tile. Preparation time and numerical error must be reported separately
from exact results.

### Paged benchmark

Run one point with `seqattn-paged-bench`, or the complete matrix with
`benchmarks/paged_sweep.py`. Results include end-to-end wall time, I/O and
queue-wait time, cache statistics, logical/physical NVMe bytes, H2D/D2H bytes,
Torch/NVML GPU memory, process RSS, and operator host-memory peaks.

The current project node is suitable for correctness, direct-I/O behavior, and
memory-limit validation only. Its NFS/local SATA storage must not be used to
claim NVMe latency. Formal runs must use a measured local device at or above
7GB/s and pass `--formal-local-nvme`; otherwise result JSON is marked as
functional/memory-only.

For pipeline testing without a physical NVMe device, `simulated-nvme` keeps
Q/K/V in caller-owned memory and applies per-page latency plus an aggregate
read/write bandwidth limit inside the normal I/O worker pipeline:

```bash
seqattn-paged-bench \
  --storage simulated-nvme \
  --tokens 61312 \
  --simulate-read-gbps 7 \
  --simulate-write-gbps 6 \
  --simulate-read-latency-us 80 \
  --simulate-write-latency-us 100 \
  --output benchmark-results/simulated_nvme_61312.json
```

The simulator is implemented separately from the real direct-I/O backend in
`seqattn.simulated_nvme`. Concurrent requests share one device bandwidth
timeline, so adding I/O workers does not multiply the configured bandwidth.
Read and write limits are independent, and fixed command latency may overlap.
Simulation results always set `storage_performance_valid=false`; they model
pipeline stalls and overlap, not filesystem, firmware, PCIe contention,
thermal behavior, or physical-device acceptance.

<p align="center">
  <img src="docs/assets/projected-pipeline-results.svg" alt="Projected pipeline benchmark results" width="100%">
</p>

## Execution pipeline

For each packed sequence and resident query super-block:

1. Copy Q to its persistent GPU buffer.
2. Stream K/V through a two- or three-slot H2D ring.
3. Run one fused Triton update per K/V tile.  The kernel reads Q from its
   resident GPU buffer and updates FP32 `(max, normalizer, accumulator)` state
   in place.
4. Prefetch the next K/V tile while the current tile computes.
5. Fuse final normalization and output casting.
6. Copy the result to pinned CPU memory while the next query super-block runs.

The planner searches 4K, 8K, and 16K K/V supertiles and jointly chooses a
resident Q chunk within the requested workspace. Its cost model includes Q
passes, repeated K/V H2D, FP32 state traffic, launch count, and copy/compute
overlap. Explicit `q_chunk_tokens` or `kv_chunk_tokens` pin either choice.
Larger query super-blocks reduce K/V rescans and PCIe traffic; smaller blocks
reduce GPU memory.

The default uses one GPU output slot because its D2H copy can already overlap
the next query super-block's attention compute; a second slot is opt-in when
profiling shows the next finalize reaches the slot before the copy completes.

## Correctness

```bash
pytest -q
```

The suite covers uneven tiles, empty packed segments, causal alignment, GQA,
cross-segment isolation, FP16/BF16 parity, and allocator stability under runner
reuse.  The PyTorch reference computes the online-softmax recurrence in FP32.

## Benchmark

Run one point per process:

```bash
seqattn-bench \
  --mode seqattn \
  --tokens 61312 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --workspace-mib 4096 --target-vram-mib 8192 \
  --kv-chunk 4096 --repeats 1 \
  --output benchmark-results/seqattn_61312.json
```

Compare against full-GPU FlashAttention 2 in an independent process:

```bash
seqattn-bench \
  --mode flash2 \
  --tokens 61312 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --target-vram-mib 8192 \
  --output benchmark-results/flash2_61312.json
```

The output includes wall time, tokens/s, effective TFLOP/s, Torch
allocated/reserved peaks, NVML process peak, planned workspace, and logical
H2D/D2H bytes.  Latency repetitions run without NVML sampling; a separate
untimed pass collects the memory peaks so driver polling cannot contaminate
short-kernel latency.  Use `scripts/profile_nsys.sh` for copy/compute overlap
and kernel timelines; profiling runs are not primary latency numbers.

The end-to-end projection benchmark compares the fused consumer path with a
staged path that materializes raw attention on the CPU:

```bash
seqattn-pipeline-bench \
  --mode pipeline \
  --tokens 61312 \
  --hidden-size 5376 --heads 56 --head-dim 128 \
  --projection-chunk 2048 --workspace-mib 2048 --kv-chunk 4096 \
  --target-vram-mib 8192 --repeats 2 \
  --output benchmark-results/pipeline_61312.json
```

See [the projected-pipeline benchmark note](docs/projected_pipeline_benchmark.md)
for the protocol, traffic accounting, and initial RTX 5090 results.

Recommended sweeps:

- sequence length: 3,072 to 61,312 tokens for MiniMax-H3 dimensions;
- workspace: 2, 4, and 6 GiB;
- KV chunk: 1,024, 2,048, 4,096, and 8,192 tokens;
- GQA ratios: 1, 4, and 8;
- causal and non-causal;
- pinned and explicitly permitted pageable input;
- `seqattn`, FlashAttention 2, and Torch SDPA baselines.

OOM and timeout results are retained.  Do not infer unmeasured maximum sequence
lengths or report profiled runs as normal latency measurements.
