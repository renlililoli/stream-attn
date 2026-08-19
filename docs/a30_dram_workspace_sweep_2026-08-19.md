# A30 DRAM workspace sweep: 400K tokens

## Scope

This report measures the current `seqattn` Ampere DRAM-streaming path as the
operator-owned HBM workspace changes. It also compares the fastest streaming
observation with a separately measured GPU-resident FlashAttention 2 baseline
for the same exact dense-attention shape.

| Parameter | Value |
|---|---:|
| GPU | NVIDIA A30 24GiB, physical GPU2 |
| Tokens / packed segments | 409,600 / 1 |
| Q heads / K heads / V heads | 56 / 56 / 56 |
| Head dimension | 128 |
| Dtype / mask | BF16 / non-causal |
| Q / K / V size | 5.469 GiB each |
| Output size | 5.469 GiB |
| Complete Q/K/V/output footprint | 21.875 GiB |
| KV chunk | 8,192 tokens |
| CPU preparation workers | 32 |
| CPU affinity | `16-31,48-63` |
| Kernel configuration | fixed: `64x64`, 4 warps, 1 stage |

Q, K, and V were held in caller-owned pinned DRAM for the streaming sweep. The
workspace value covers buffers planned and allocated by `seqattn`; CUDA context
and other process-level allocations account for the difference between the
workspace and PID-level NVML peak.

## Measurement protocol

All workspace points ran serially in one benchmark process on one physical
GPU. There was no concurrent SeqAttn scan. GPU2 was idle before the run, but the
host was not globally exclusive.

Input preparation ran once with 32 CPU workers and took 24.7565 seconds
(0.6627 GiB/s). The execution column excludes this preparation phase. Each
workspace has one observation, so the table reports measurements directly
rather than means, medians, or error bars.

The GPU-resident baseline came from an independent 2026-08-18 process using
FlashAttention 2 `2.7.4.post1`. It used the same shape, dtype, mask, seed, and
sample positions. The optimized-kernel container used for the streaming sweep
did not include FlashAttention 2, so the baseline was not rerun.

## Results

| HBM workspace | PID GPU peak | Resident Q | Q passes | Logical H2D | Execution | Tokens/s | Effective TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 GiB | 0.717 GiB | 576 | 712 | 7,793 GiB | 681.162 s | 601 | 7.06 |
| **1 GiB** | **1.229 GiB** | **9,856** | **42** | **465 GiB** | **102.319 s** | **4,003** | **47.01** |
| 2 GiB | 2.221 GiB | 28,416 | 15 | 170 GiB | 105.233 s | 3,892 | 45.71 |
| 4 GiB | 4.217 GiB | 65,600 | 7 | 82 GiB | 106.956 s | 3,830 | 44.98 |
| 6 GiB | 6.213 GiB | 102,720 | 4 | 49 GiB | 108.102 s | 3,789 | 44.50 |
| 8 GiB | 8.213 GiB | 139,904 | 3 | 38 GiB | 108.696 s | 3,768 | 44.26 |
| 12 GiB | 12.213 GiB | 214,208 | 2 | 27 GiB | 109.078 s | 3,755 | 44.10 |
| 16 GiB | 16.213 GiB | 288,512 | 2 | 27 GiB | 108.820 s | 3,764 | 44.20 |

## FlashAttention 2 baseline

| Backend | Q/K/V residency | Torch GPU peak | Execution | Effective TFLOPS | Relative latency |
|---|---|---:|---:|---:|---:|
| FlashAttention 2 | GPU HBM | 21.961 GiB | 50.827 s | 94.6 | 1.000x |
| **seqattn, 1GiB workspace** | **Pinned CPU DRAM** | **0.968 GiB** | **102.319 s** | **47.01** | **2.013x** |

The comparison uses torch peak allocated memory for both rows. The streaming
row's PID-level NVML peak is 1.229GiB after including CUDA context and other
process allocations. FlashAttention 2 is the lower-latency path when the full
21.875GiB working set fits in HBM. The 1GiB seqattn point reduces torch peak
allocated memory by 95.6%, while execution takes 2.013x as long because K/V is
streamed repeatedly over PCIe. FlashAttention 2 leaves output in HBM, while the
seqattn execution time includes the final 5.469GiB D2H output transfer. Neither
execution time includes input preparation; the FlashAttention 2 measurement
also excludes its separately reported 1.440-second HBM-residency preparation.

## Interpretation

The meaningful capacity boundary is between 0.5GiB and 1GiB. Increasing the
workspace to 1GiB raises resident Q from 576 to 9,856 tokens, reduces Q passes
from 712 to 42, and cuts logical H2D from 7.61TiB to 465GiB. Execution falls
from 681.162 seconds to 102.319 seconds.

Every observation from 1GiB through 16GiB is within 6.61% of the fastest point.
Larger workspaces continue to reduce Q passes and H2D traffic, but do not
improve wall time on this A30. The fixed `64x64/4/1` profile improves the 1GiB
point by 3.90% over the earlier `64x64/4/2` observation. The improvement varies
from 0.04% to 3.90% across the complete sweep.

This makes 1GiB the throughput-oriented setting for the tested shape. Larger
workspaces are useful only when reducing PCIe traffic or host DRAM contention
is more important than reclaiming HBM for the rest of the model.

## Numerical consistency

The benchmark sampled eight BF16 values at each of five token positions. All
eight workspace points produced identical sampled signatures. Compared with
the GPU-resident FlashAttention 2 signature across the same 40 values:

```text
relative L2:  0.004160
max absolute: 3.052e-5
cosine:       0.99999219
```

These are sparse signature metrics, not full-output error norms.

## Limits and artifacts

- Each workspace value has one observation.
- The scan used one GPU and one SeqAttn process; no multi-GPU throughput claim
  is implied.
- CPU affinity was fixed, but the complete host was not reserved exclusively.
- Data preparation is reported separately and excluded from execution time.
- The test exercises caller-owned DRAM streaming, not the paged or NVMe path.
- The FlashAttention 2 baseline is from the previous day and a separate
  container, though the model shape and deterministic input configuration are
  identical.

The source-of-truth JSON files are retained locally at:

```text
workspace/benchmarks/artifacts/a30_seqattn_400k_optimized_20260819/
  a30-workspace-sweep-400k-s1.json
workspace/benchmarks/artifacts/a30_seqattn_400k_20260818/
  a30-gpu-resident-400k.json
```
