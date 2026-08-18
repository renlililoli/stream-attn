# A30 large attention storage-tier benchmark

## Scope

This benchmark re-runs the 524K-token storage-tier experiment on one NVIDIA
A30. The A30 has 24GiB of HBM, so the fixed shape uses 409,600 tokens: the
complete GPU-resident working set is 21.9GiB and just fits the card.

```text
GPU:                 NVIDIA A30, 24GiB, Ampere sm_80, PCIe 4.0 x16
tokens:              409,600
packed segments:     1
Q/K/V heads:         56 / 56 / 56
head dimension:      128
dtype:               BF16
causal:              false
Q bytes:             5.47 GiB
K bytes:             5.47 GiB
V bytes:             5.47 GiB
output bytes:        5.47 GiB
Q/K/V total:         16.4 GiB
GPU-resident total:  21.9 GiB
```

GPU-resident execution uses FlashAttention 2 (`flash-attn 2.7.4.post1`).
CPU-backed execution uses the seqattn Triton online-softmax kernel. All runs
use the same deterministic input and the same output signature locations.

```text
container:      pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime + gcc + numactl
torch:          2.7.0+cu126
triton:         3.3.0
driver:         580.173.02
CPU binding:    numactl --physcpubind=0-47 (NUMA node 0, both A30s)
data prep:      make_tensors_parallel, 32 workers, 4096-token chunks, 21.6 s
```

NVML per-process sampling is unavailable in this container (the installed
pynvml predates driver 580 and reports zero), so GPU peaks are torch
allocator peaks. All rows are single observations on one idle GPU; there are
no error bars and no cross-GPU medians, unlike the 5090 scan.

## A30 results

| Mode | Execution | GPU peak (torch) | Effective TFLOPS | Status |
|---|---:|---:|---:|---|
| FlashAttention 2 GPU resident | 50.827 s | 21.96 GiB | 94.6 | success |

The torch peak of 21.96GiB on a 24GiB card confirms the working set was sized
to just fit. H2D residency preparation took 1.44s; it is reported separately
and not included in execution time.

## DRAM workspace sweep

Complete Q/K/V stays in unrestricted caller-owned pinned DRAM and only the
seqattn HBM workspace varies:

```text
0.5 / 1 / 2 / 4 / 6 / 8 / 12 / 16 GiB
```

Each row records the planner-selected Q chunk, actual GPU peak, execution
time, effective TFLOPS, H2D/D2H traffic, process RSS, and the output signature.
The scan uses `StreamingAttentionRunner`, not the fixed-host-budget paged
cache, so the curve isolates the HBM-resident-Q versus repeated-DRAM-streaming
tradeoff.

| HBM | Execution | GPU peak | Q chunk | Q passes | H2D | D2H | TFLOPS | vs resident |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 GiB | 681.446 s | 0.47 GiB | 576 | 712 | 7.6 TiB | 5.5 GiB | 7.1 | 13.407x |
| 1 GiB | 106.476 s | 0.97 GiB | 9,856 | 42 | 465 GiB | 5.5 GiB | 45.2 | 2.095x |
| 2 GiB | 106.293 s | 1.97 GiB | 28,416 | 15 | 170 GiB | 5.5 GiB | 45.3 | 2.091x |
| 4 GiB | 107.761 s | 3.97 GiB | 65,600 | 7 | 82 GiB | 5.5 GiB | 44.6 | 2.120x |
| 6 GiB | 108.331 s | 5.97 GiB | 102,720 | 4 | 49 GiB | 5.5 GiB | 44.4 | 2.131x |
| 8 GiB | 109.669 s | 7.97 GiB | 139,904 | 3 | 38 GiB | 5.5 GiB | 43.9 | 2.158x |
| 12 GiB | 110.911 s | 11.97 GiB | 214,208 | 2 | 27 GiB | 5.5 GiB | 43.4 | 2.182x |
| 16 GiB | 111.170 s | 15.97 GiB | 288,512 | 2 | 27 GiB | 5.5 GiB | 43.3 | 2.187x |

Process RSS peak was 32.7GiB for every row: 21.9GiB of caller-owned pinned
Q/K/V plus the 5.5GiB pinned output, none of which is inside any operator
budget.

### Interpretation

The A30 curve has three regions. Between 1GiB and 2GiB the operator is
compute-bound at about 45 TFLOPS and execution is flat (106.3-106.5s); the
two points differ by 0.17%. Above 2GiB wall time rises monotonically with the
workspace to 111.2s at 16GiB (+4.6%): larger Q chunks grow the per-K/V-tile
FP32 state traffic inside the kernel, and removing H2D rescans does not pay
for itself on this GPU. At 0.5GiB the curve falls off a cliff: the planner
drops to a 576-token Q chunk, Q passes jump to 712, H2D reaches 7.6TiB, and
execution degrades 6.4x to 681.4s as the operator becomes PCIe-bound.

This is the opposite of the 5090 result, where 4GiB to 8GiB improved
execution by 5.3% before a plateau. On the A30 the streaming path has a
1-2GiB sweet spot, stays within 4.6% of it up to 16GiB, and needs roughly
1GiB of HBM workspace before repeated K/V streaming dominates the wall time.

The defensible statement for this shape is: at 400K tokens on an A30, the
operator is compute-bound from 1GiB upward with a 1-2GiB optimum, collapses
below 1GiB, and runs 2.09-2.19x slower than the 21.9GiB GPU-resident
FlashAttention 2 path inside the feasible region.

## Numerical sampling

All five sweep output signatures are identical. Across the 40 sampled BF16
values, compared with GPU-resident FlashAttention 2:

```text
relative L2:  0.004160
max absolute: 3.052e-5
cosine:       0.99999219
```

These are sparse signature metrics, not full-output error norms.

## Result artifacts

The JSON checkpoints containing the raw measurements are:

```text
workspace/benchmarks/artifacts/a30_seqattn_400k_20260818/
  a30-gpu-resident-400k.json
  a30-workspace-sweep-400k.json
```

The benchmark scripts checkpoint after every workspace so a completed point is
retained if a later point is interrupted.
