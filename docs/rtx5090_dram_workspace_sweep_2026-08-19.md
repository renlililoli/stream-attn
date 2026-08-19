# RTX 5090 DRAM workspace sweep: 524K tokens

## Scope

This report measures the current `seqattn` Blackwell DRAM-streaming path as the
operator-owned HBM workspace changes. It is a capacity and execution-time
measurement of one exact dense-attention shape, not a comparison with an older
kernel or another implementation.

| Parameter | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 5090, physical GPU3 |
| Tokens / packed segments | 524,288 / 1 |
| Q heads / K heads / V heads | 56 / 56 / 56 |
| Head dimension | 128 |
| Dtype / mask | BF16 / non-causal |
| Q / K / V size | 7 GiB each |
| Output size | 7 GiB |
| Complete Q/K/V/output footprint | 28 GiB |
| KV chunk | 8,192 tokens |
| CPU preparation workers | 32 |
| CPU affinity | `160-191,416-447` |
| Kernel configuration | automatic: `128x64`, 8 warps, 3 stages |

Q, K, and V were held in caller-owned pinned DRAM. The workspace value covers
the buffers planned and allocated by `seqattn`; CUDA context and other
process-level allocations account for the difference between workspace and
PID-level NVML peak.

## Measurement protocol

All workspace points ran serially in one benchmark process on one physical GPU.
There was no concurrent SeqAttn scan, which avoids competition between two GPU
pipelines for host DRAM bandwidth. GPU3 was idle before the run, but the host
was not globally exclusive.

Input preparation ran once with 32 CPU workers and took 25.8366 seconds
(0.8128 GiB/s). The execution column excludes this preparation phase. Each
workspace has one observation; the table therefore reports measurements
directly rather than means, medians, or error bars.

## Results

| HBM workspace | PID GPU peak | Resident Q | Q passes | Logical H2D | Execution | Tokens/s | Effective TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 1.541 GiB | 9,856 | 54 | 763 GiB | 38.201 s | 13,724 | 206.3 |
| 2 GiB | 2.533 GiB | 28,416 | 19 | 273 GiB | 36.247 s | 14,464 | 217.4 |
| 4 GiB | 4.520 GiB | 65,536 | 8 | 119 GiB | 36.073 s | 14,534 | 218.5 |
| 6 GiB | 6.520 GiB | 102,656 | 6 | 91 GiB | 36.122 s | 14,514 | 218.2 |
| 8 GiB | 8.525 GiB | 139,904 | 4 | 63 GiB | 36.113 s | 14,518 | 218.2 |
| 10 GiB | 10.525 GiB | 177,024 | 3 | 49 GiB | 36.111 s | 14,519 | 218.3 |
| 12 GiB | 12.520 GiB | 214,144 | 3 | 49 GiB | 36.053 s | 14,542 | 218.6 |
| 14 GiB | 14.525 GiB | 251,392 | 3 | 49 GiB | 36.010 s | 14,559 | 218.9 |

## Interpretation

The meaningful latency transition occurs between 1GiB and 2GiB. Increasing
the workspace to 2GiB reduces execution time by 5.1%, from 38.201 seconds to
36.247 seconds. Every observation from 2GiB through 14GiB is within 0.66% of
the fastest point.

The 4-14GiB observations span 36.010-36.122 seconds, a 0.31% range. More HBM
still improves data movement: Q passes fall from 8 at 4GiB to 3 at 14GiB, and
logical H2D falls from 119GiB to 49GiB. For this shape and system, those traffic
reductions do not translate into a material wall-time reduction once the
2GiB workspace has reached the throughput plateau.

This result makes 2GiB a practical throughput-oriented setting for the tested
shape. Larger workspaces remain useful when reducing PCIe traffic, leaving CPU
DRAM bandwidth available to other work, or reducing the number of launches is
more important than reclaiming HBM for the rest of the model.

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

The source-of-truth JSON is retained locally at:

```text
workspace/benchmarks/results/rtx5090_dram_workspace_524k_optimized_20260819/gpu3_auto_blackwell_single.json
```
