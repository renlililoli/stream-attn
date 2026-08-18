# seqattn

`seqattn` is an inference-only, exact attention library for sequences whose
Q/K/V tensors live in CPU DRAM.  It uses GPU memory as a statically planned
resident working set: query super-blocks stay resident while K/V tiles stream
through a small ring buffer.  The attention result is produced without
materializing the score matrix or the complete Q/K/V set on the GPU.

The project follows the same IO-aware principle that makes FlashAttention
effective inside the GPU memory hierarchy, but applies it one level higher:

```text
FlashAttention:   HBM is the backing store, SRAM/registers are the working set
seqattn:          CPU DRAM is the backing store, HBM is the working set
```

This is not a replacement for FlashAttention when full Q/K/V fit in VRAM.
It targets exact long-context inference under a fixed GPU-memory budget and
reports the resulting PCIe and latency cost explicitly.

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

The planner maximizes resident query tokens within the requested workspace.
Larger query super-blocks reduce K/V rescans and PCIe traffic; smaller blocks
reduce GPU memory.  This is the main memory/traffic tradeoff exposed by the
library.

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
