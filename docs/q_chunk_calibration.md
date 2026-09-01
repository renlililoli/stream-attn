# Calibrating SeqAttn Chunk Sizes

This guide selects `q_chunk_tokens` from measured hardware behavior rather
than from a generic GPU model name or a workspace-size heuristic. The same GPU
can require a different Q chunk when CPU memory placement, NUMA affinity,
FlashAttention backend, dtype, head layout, or system load changes.

## Parameter priority

| Parameter | Role | Recommended treatment |
|---|---|---|
| `q_chunk_tokens` | HBM-resident query super-block | Calibrate from the host-memory roofline on each deployment topology |
| `kv_chunk_tokens` | Streamed K/V transfer and update tile | Hold at `4096` during Q calibration; sweep only after Q is fixed |
| `projection_tile_tokens` | QKV projection tile | Use the integration default first; tune with a real block microbenchmark |
| `ffn_tile_tokens` | MLP activation tile | Use the integration default first; increase only when measured gains justify the extra HBM |

`q_chunk_tokens` is normally the important performance choice. It controls how
many times the complete K/V sequence must be streamed from pinned host memory.
The other three values affect launch amortization, overlap, and transient
workspace, but usually produce smaller performance changes once their kernels
are saturated.

## Roofline model

Measure two independent quantities with the final backend and topology:

- `B`: concurrent pinned-host-to-GPU K/V bandwidth in decimal GB/s while the
  partial-attention kernel is active;
- `P`: effective throughput in TFLOP/s for the matching GPU-resident attention
  backend and head shape.

Do not substitute theoretical PCIe bandwidth or the GPU vendor's peak tensor
TFLOPS. Both omit the actual memory population, NUMA placement, copy/kernel
overlap, backend efficiency, and shape-dependent kernel saturation.

For BF16/FP16 MHA, the first-order streaming roof is:

```text
P_stream(q_effective) = min(B_GBps * q_effective / 1000, P_TFLOPS)
q_star                = 1000 * P_TFLOPS / B_GBps
```

For GQA with `group_size = q_heads / kv_heads`, multiply the bandwidth branch
by `group_size`, so:

```text
q_star = 1000 * P_TFLOPS / (B_GBps * group_size)
```

This is a prospective knee, not the final setting. SeqAttn and the resident
backend can have slightly different compute plateaus, and finite sequences
produce a staircase:

```text
q_passes          = ceil(sequence_tokens / q_chunk_tokens)
q_effective       = sequence_tokens / q_passes
```

Use `q_effective`, not only the requested Q value, when comparing the model to
measurements.

## Calibration procedure

### 1. Freeze the deployment topology

Use the same physical GPU, CPU affinity, NUMA memory policy, dtype, head shape,
backend, K/V tile, and container image that production will use. Ensure the GPU
and relevant host-memory channels are otherwise idle. Record `nvidia-smi`, CPU
affinity, and `numactl --hardware` output.

NUMA policy is part of the result. For example, the same RTX 5090 measured
about 37.28 GB/s with pinned pages on one populated memory node and 56.72 GB/s
when pages were interleaved across two populated nodes. Those policies produced
different Q knees.

### 2. Measure concurrent H2D bandwidth

Keep the production K/V tile fixed. The calibration below reports bare copies
and copies concurrent with representative partial-attention compute:

```bash
PYTHONPATH=src python benchmarks/host_memory_roofline_calibration.py \
  --mode both \
  --kv-chunk 4096 \
  --compute-q 8192 16384 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --dtype bfloat16 \
  --warmup 10 --repeats 50 \
  --output results/roofline/h2d.json
```

Run the command under the intended `numactl --membind`, `--interleave`, and CPU
binding policy. Use the median concurrent copy rate, not the bare-copy maximum.
Repeat if the p10-p90 range is wide or another process used the GPU or memory
channels during the run.

### 3. Measure the resident backend roof

Use the same attention backend, dtype, heads, head dimension, causal mode, and
a sequence large enough to saturate the kernel while still fitting in HBM:

```bash
PYTHONPATH=src python -m seqattn_core.benchmarking.resident \
  --backend flash4 \
  --tokens 524288 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --dtype bfloat16 \
  --warmup 5 --repeats 10 \
  --output results/roofline/resident.json
```

Use the median CUDA-event effective throughput as `P`. Select `flash2` or
`flash4` according to the backend used by the deployment. If the production
sequence does not fit as a resident benchmark, use the largest representative
shape that reaches a stable backend plateau and document that substitution.

### 4. Predict and align the knee

Compute `q_star`, then align it to the backend's query block, normally 128
tokens. Check the SeqAttn plan's estimated workspace before running. For the
recorded RTX 5090 FA4 shape:

| Memory policy | `B` | `P` | Predicted knee | Selected requested Q |
|---|---:|---:|---:|---:|
| One populated NUMA memory node | 37.284 GB/s | 213.323 TFLOP/s | 5,721.56 | `5760` |
| Two populated nodes interleaved | 56.717 GB/s | 213.323 TFLOP/s | 3,761.18 | `3840` |

The smaller Q for the faster host-memory policy is expected: higher bandwidth
needs less resident-query reuse to hide K/V transfers.

### 5. Validate around the prediction

Run each candidate in a fresh process. Start with approximately `0.75x`,
`0.9x`, `1.0x`, `1.1x`, and `1.25x` of the aligned prediction, using one warmup
and at least three measured repeats:

```bash
PYTHONPATH=src python -m seqattn_core.benchmarking.streaming \
  --mode seqattn \
  --tokens 524288 --segments 1 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --dtype bfloat16 \
  --q-chunk 5760 --kv-chunk 4096 \
  --workspace-mib 4096 \
  --warmup 1 --repeats 3 \
  --skip-memory-probe \
  --output results/roofline/q_05760.json
```

Select the smallest aligned requested Q whose median pipeline throughput is at
least 95% of the clean local plateau and whose integration-level process VRAM
peak stays within the deployment target. Re-run the selected point in the real
model because checkpoint kernels, weight staging, and non-attention
activations are outside the standalone operator benchmark.

## Secondary tile tuning

Keep `kv_chunk_tokens=4096` while calibrating Q. Changing K/V tile size changes
both the copy payload and partial-attention kernel shape, so a material K/V
change requires repeating the concurrent-bandwidth and Q-knee validation.
Consider a `2048, 4096, 8192` K/V sweep only when the default does not overlap
copies effectively or the target sequence is relatively short.

Tune QKV projection and MLP tiles only after Q and K/V are fixed. Use a real
single-block benchmark with production quantized linear operators, one warmup,
multiple steady repeats, and peak-memory recording. For the measured MiniMax-H3
INT8 ConvRot block on RTX 5090, `4096` was the QKV projection sweet point.
Increasing MLP tiles beyond `4096` produced comparatively small latency gains
while increasing transient HBM, so the integration keeps both defaults at
`4096`. These are measured defaults, not universal constants.

## When to recalibrate

Recalibrate Q after changing any of the following:

- GPU architecture or attention backend;
- CPU/NUMA affinity, memory interleave policy, or memory-channel population;
- K/V chunk, dtype, head count, head dimension, or MHA/GQA layout;
- PCIe topology, virtualization, container CPU set, or concurrent system load.

The complete reference experiments are the
[RTX 5090 roofline report](rtx5090_host_memory_roofline_experiment0_2026-08-24.md)
and the [A30 roofline report](a30_host_memory_roofline_experiment0_2026-08-24.md).
