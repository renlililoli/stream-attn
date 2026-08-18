# seqattn

`seqattn` is an inference-only, exact attention library for sequences whose
Q/K/V tensors live in CPU DRAM.  It treats GPU memory as a managed cache:
query super-blocks stay resident while K/V tiles stream through a small ring
buffer.  The attention result is produced without materializing the score
matrix or the complete Q/K/V set on the GPU.

The project follows the same IO-aware principle that makes FlashAttention
effective inside the GPU memory hierarchy, but applies it one level higher:

```text
FlashAttention:   HBM is the backing store, SRAM/registers are the cache
seqattn:          CPU DRAM is the backing store, HBM is the cache
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

`workspace_budget_bytes` only covers buffers owned by `seqattn`.  Whole-process
budgets must also reserve memory for the CUDA context, weights, and caller-owned
activations.

## Execution pipeline

For each packed sequence and resident query super-block:

1. Copy Q to its persistent GPU buffer.
2. Stream K/V through a two- or three-slot H2D ring.
3. Run one fused Triton update per K/V tile.  The kernel holds Q in the GPU
   cache and updates FP32 `(max, normalizer, accumulator)` state in place.
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
