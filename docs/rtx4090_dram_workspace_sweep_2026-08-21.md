# RTX 4090 DRAM workspace sweep: 524K tokens

## Scope

This report measures the current `seqattn` Ada DRAM-streaming path as the
operator-owned HBM workspace changes. It repeats the RTX 5090 workspace
protocol on one physical RTX 4090. The exact shape cannot be paired with a
GPU-resident FlashAttention observation because the complete working set does
not fit in this card's HBM.

| Parameter | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4090, physical GPU0 |
| Tokens / packed segments | 524,288 / 1 |
| Q heads / K heads / V heads | 56 / 56 / 56 |
| Head dimension | 128 |
| Dtype / mask | BF16 / non-causal |
| Q / K / V size | 7 GiB each |
| Output size | 7 GiB |
| Complete Q/K/V/output footprint | 28 GiB |
| KV chunk | 8,192 tokens |
| CPU preparation workers | 32 |
| CPU affinity | `0-15,32-47` |
| Kernel configuration | automatic: `64x64`, 4 warps, 2 stages |

Q, K, and V were held in caller-owned pinned DRAM. The workspace value covers
the buffers planned and allocated by `seqattn`; CUDA context and other
process-level allocations account for the difference between workspace and
PID-level NVML peak.

## Measurement protocol

All workspace points ran serially in one benchmark process on one physical
GPU. There was no concurrent SeqAttn scan, which avoids competition between two
GPU pipelines for host DRAM bandwidth. GPU0 was idle before the run, but the
host was not globally exclusive.

Input preparation ran once with 32 CPU workers and took 33.2228 seconds
(0.6321 GiB/s). The execution column excludes this preparation phase. Each
workspace has one observation; the table therefore reports measurements
directly rather than means, medians, or error bars.

The exact 28 GiB Q/K/V/output shape exceeds the RTX 4090's 24,564 MiB device
capacity. A same-shape GPU-resident FlashAttention baseline was therefore not
run; a smaller resident shape would answer a different question.

## Results

| HBM workspace | PID GPU peak | Resident Q | Q passes | Logical H2D | Execution | Tokens/s | Effective TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 1.385 GiB | 9,856 | 54 | 763 GiB | 107.564 s | 4,874 | 73.3 |
| **2 GiB** | **2.377 GiB** | **28,416** | **19** | **273 GiB** | **67.076 s** | **7,816** | **117.5** |
| 4 GiB | 4.373 GiB | 65,600 | 8 | 119 GiB | 66.552 s | 7,878 | 118.4 |
| 6 GiB | 6.369 GiB | 102,720 | 6 | 91 GiB | 66.982 s | 7,827 | 117.7 |
| 8 GiB | 8.369 GiB | 139,904 | 4 | 63 GiB | 66.504 s | 7,884 | 118.5 |
| 10 GiB | 10.369 GiB | 177,024 | 3 | 49 GiB | 66.691 s | 7,861 | 118.2 |
| 12 GiB | 12.369 GiB | 214,208 | 3 | 49 GiB | 66.506 s | 7,883 | 118.5 |
| 14 GiB | 14.369 GiB | 251,392 | 3 | 49 GiB | 66.296 s | 7,908 | 118.9 |

The automatic Ada launch profile resolves to `64x64`, 4 warps, and 2 stages at
every point. Moving from 1GiB to 2GiB reduces execution time by 37.64% and
raises observed throughput by 1.60x. The complete 2-14GiB range stays within
1.18% of the fastest observation, while larger workspaces continue to reduce Q
passes and logical H2D traffic. All eight sampled output signatures are
identical. Data preparation is excluded from execution time and took 33.223
seconds with 32 CPU workers.

## Interpretation

The meaningful latency transition occurs between 1GiB and 2GiB. Increasing the
workspace to 2GiB reduces execution time from 107.564 seconds to 67.076
seconds. Every observation from 2GiB through 14GiB is within 1.18% of the
fastest point.

The 2-14GiB observations span 66.296-67.076 seconds. More HBM still improves
data movement: Q passes fall from 19 at 2GiB to 3 at 10-14GiB, and logical H2D
falls from 273GiB to 49GiB. For this shape and system, those traffic reductions
do not translate into a material wall-time reduction once the 2GiB workspace
has reached the throughput plateau.

This makes 2GiB a practical throughput-oriented setting for the tested shape.
Larger workspaces remain useful when reducing PCIe traffic, leaving CPU DRAM
bandwidth available to other work, or reducing the number of launches is more
important than reclaiming HBM for the rest of the model.

## Numerical consistency

The benchmark sampled eight BF16 values at each of five token positions:
`0`, `131072`, `262144`, `393216`, and `524287`. All eight workspace points
produced identical sampled signatures. This is a consistency check across
planner choices, not a substitute for the repository's full reference-based
correctness tests.

## Limits and artifacts

- Each workspace value has one observation.
- The scan used one GPU and one SeqAttn process; no multi-GPU throughput claim
  is implied.
- CPU affinity was fixed, but the complete host was not reserved exclusively.
- Data preparation is reported separately and excluded from execution time.
- The test exercises caller-owned DRAM streaming, not the paged or NVMe path.
- The exact-shape resident baseline is unavailable because 28 GiB does not fit
  in 24,564 MiB of HBM.
- The first Python 3.10 launch failed on `typing.Self`; the formal benchmark
  used Python 3.12 without changing source code, exposing a mismatch between
  the declared Python 3.10 floor and the current import surface.

The source-of-truth JSON and local analysis bundle are retained at:

```text
benchmark-results/rtx4090-dram-workspace-2026-08-21/
  workspace_sweep.json
```
