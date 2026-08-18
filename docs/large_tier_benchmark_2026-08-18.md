# Large attention storage-tier benchmark

## Scope

This benchmark was run on August 18, 2026 to compare three execution modes for
one dense BF16 attention sequence whose activation footprint satisfies all of
the following constraints:

- complete Q/K/V plus output fits in a 32GiB GPU;
- a 2GiB HBM workspace plus unrestricted CPU DRAM can execute the operator;
- a 2GiB HBM workspace plus an 8GiB operator-owned host budget cannot cache the
  complete K/V working set, forcing repeated backing-store reads.

The fixed shape is:

```text
tokens:             524,288
packed segments:    1
Q/K/V heads:        56 / 56 / 56
head dimension:     128
dtype:              BF16
Q bytes:            7 GiB
K bytes:            7 GiB
V bytes:            7 GiB
output bytes:       7 GiB
Q/K/V total:        21 GiB
GPU-resident total: 28 GiB
```

GPU-resident execution uses FlashAttention 2. CPU-backed execution uses the
seqattn Triton online-softmax kernel. Simulated NVMe uses 7 GB/s aggregate read
bandwidth, 6 GB/s aggregate write bandwidth, 80 microsecond read latency,
100 microsecond write latency, and queue depth four.

## GPU3 exclusive-process results

Physical GPU3 was configured in `Exclusive_Process` mode. All rows below use
the same deterministic input and output signature locations.

| Mode | Run 1 | Run 2 | Two-run median | GPU peak |
|---|---:|---:|---:|---:|
| GPU resident | 38.137 s | 38.170 s | 38.154 s | 28.707 GiB |
| 2GiB HBM + 8GiB operator DRAM | 196.234 s | 226.273 s | 211.254 s | 2.533 GiB |
| 2GiB HBM + simulated NVMe | 204.720 s | 105.685 s | 155.202 s | 2.533 GiB |

The GPU-resident runs differ by only 0.087%. The host-backed runs remain much
more variable despite exclusive GPU access, showing that GPU exclusivity does
not isolate CPU scheduling or host-memory bandwidth.

The fixed-budget paged paths report:

```text
operator host peak:  7.491 GiB
pinned peak:         0.998 GiB
DRAM K/V cache:      6.368 GiB
K/V working set:     14 GiB
cache hit ratio:     35.05%
K/V page scans:      32
cache misses:        9,477
H2D traffic:         455 GiB
D2H traffic:         7 GiB
process RSS peak:    approximately 32.2 GiB
```

Process RSS is larger than the operator budget because the in-memory simulator
retains the caller-owned 21GiB Q/K/V backing tensors. Those caller-owned
tensors are intentionally outside `HostMemoryPlan`. A physical NVMe source
would not retain them in process DRAM.

## Simulated NVMe accounting

Both simulated runs use the same deterministic cache trace and service model:

```text
simulated logical reads:  297.654 GiB
simulated logical writes: 7 GiB
read service time:        46.417 s
write service time:       1.254 s
```

The read service time matches payload divided by configured bandwidth plus per
page latency. End-to-end variation comes from host page copying, cache lookup,
thread-pool queueing, H2D supply, and GPU stream stalls rather than a change in
the configured simulated device.

| Metric | NVMe run 1 | NVMe run 2 |
|---|---:|---:|
| End-to-end execution | 204.720 s | 105.685 s |
| Simulated read elapsed | 71.454 s | 53.202 s |
| Simulated read queue | 24.129 s | 5.949 s |
| Runtime I/O queue wait | 169.828 s | 55.197 s |
| First-to-last CUDA event span | 202.006 s | 103.152 s |

The two-run NVMe median is not a stable performance conclusion. Additional
host isolation and repeats are required before interpreting NVMe/DRAM ratios.
These results remain timing simulations and are not physical NVMe acceptance
measurements.

## Numerical sampling

The DRAM and simulated-NVMe output signatures are identical. Across 40 sampled
BF16 output values, compared with GPU-resident FlashAttention 2:

```text
relative L2:  0.002958
max absolute: 3.0518e-5
cosine:       0.99999586
```

These are sparse signature metrics, not full-output error norms.

## DRAM workspace sweep

The next experiment keeps complete Q/K/V in unrestricted caller-owned DRAM and
varies only the seqattn HBM workspace:

```text
4 / 6 / 8 / 10 / 12 / 16 GiB
```

Each point records the planner-selected Q chunk, actual GPU peak, execution
time, effective TFLOPS, H2D/D2H traffic, process RSS, and output signature. The
scan uses `StreamingAttentionRunner`, not the fixed-host-budget paged cache, so
the resulting curve isolates the HBM-resident-Q versus repeated-DRAM-streaming
tradeoff.

Two cross-assigned scans ran concurrently on otherwise idle GPU2 and exclusive
GPU3. Each process was bound to the CPU affinity reported by `nvidia-smi topo
-m` so pinned-memory first-touch was separated as far as the machine topology
allowed:

```text
scan 1:
  GPU2 CPU affinity: 192-223,448-479; workspaces 4/8/12 GiB
  GPU3 CPU affinity: 160-191,416-447; workspaces 6/10/16 GiB
scan 2:
  GPU2 CPU affinity: 192-223,448-479; workspaces 6/10/16 GiB
  GPU3 CPU affinity: 160-191,416-447; workspaces 4/8/12 GiB
```

| HBM | GPU2 | GPU3 | Median | Span | GPU peak | Q chunk | Q passes | H2D | Median TFLOPS | vs resident |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 GiB | 54.235 s | 52.391 s | 53.313 s | 3.46% | 4.529 GiB | 65,600 | 8 | 119 GiB | 147.87 | 1.397x |
| 6 GiB | 51.547 s | 53.258 s | 52.403 s | 3.26% | 6.525 GiB | 102,720 | 6 | 91 GiB | 150.44 | 1.373x |
| 8 GiB | 50.044 s | 50.936 s | 50.490 s | 1.77% | 8.525 GiB | 139,904 | 4 | 63 GiB | 156.11 | 1.323x |
| 10 GiB | 50.013 s | 51.030 s | 50.521 s | 2.01% | 10.525 GiB | 177,024 | 3 | 49 GiB | 156.02 | 1.324x |
| 12 GiB | 50.039 s | 50.956 s | 50.498 s | 1.81% | 12.525 GiB | 214,208 | 3 | 49 GiB | 156.09 | 1.324x |
| 16 GiB | 49.950 s | 50.992 s | 50.471 s | 2.07% | 16.525 GiB | 288,512 | 2 | 35 GiB | 156.17 | 1.323x |

All six output signatures are identical to the fixed-budget DRAM and simulated
NVMe signatures. The actual GPU peak is approximately 0.525GiB above the
operator workspace estimate because the CUDA context and non-workspace process
allocations are visible to NVML.

The primary gain occurs between 4GiB and 8GiB: Q passes fall from eight to four,
H2D falls from 119GiB to 63GiB, and median execution improves by 5.3%. Above
8GiB, all medians are within 0.10%, while the same-workspace GPU2/GPU3 span is
1.77% to 2.07%. Reducing Q passes from four to two lowers H2D to 35GiB but does
not improve wall time. This indicates that this 524,288-token point is no
longer dominated by repeated H2D once an approximately 8GiB workspace is
available.

The two measurements at each workspace are cross-GPU observations collected
while another process was concurrently exercising the other GPU and its local
DRAM path. They are not isolated same-GPU repeats. Consequently, the precise
ordering within the 8/10/12/16GiB plateau is below the observed system noise
and must not be interpreted as a workspace optimum. The defensible result is
that 8GiB captures essentially all of the measured benefit over 4GiB for this
shape, while remaining about 1.32x slower than the 28GiB GPU-resident path.

## Result artifacts

The JSON checkpoints containing the raw measurements are:

```text
workspace/benchmarks/artifacts/seqattn_gpu3_mha_524288_20260818/
workspace/benchmarks/artifacts/seqattn_gpu2_dram_workspace_524288_20260818/
workspace/benchmarks/artifacts/seqattn_gpu3_dram_workspace_524288_20260818/
```

The benchmark scripts checkpoint after every workspace so a completed point is
retained if a later point is interrupted.
