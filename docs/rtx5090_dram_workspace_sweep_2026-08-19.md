# RTX 5090 DRAM workspace sweep: 524K tokens

## Scope

This report measures the current `seqattn` Blackwell DRAM-streaming path as the
operator-owned HBM workspace changes. It also places the sweep next to
GPU-resident FlashAttention 2 and FlashAttention 4 observations of the same
exact dense-attention shape. It is not a comparison with an older seqattn
kernel.

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

All original 1-14GiB workspace points ran serially in one benchmark process on
one physical GPU. There was no concurrent SeqAttn scan, which avoids
competition between two GPU pipelines for host DRAM bandwidth. GPU3 was idle
before the run, but the host was not globally exclusive.

Input preparation ran once with 32 CPU workers and took 25.8366 seconds
(0.8128 GiB/s). The execution column excludes this preparation phase. Each
workspace has one observation; the table therefore reports measurements
directly rather than means, medians, or error bars.

The GPU-resident measurements ran afterward in independent processes on the
same physical GPU. Each used one full-shape warmup followed by one measured
execution. Deterministic Q/K/V generation used the same seed, 32-worker
4096-token chunking rule, dtype, shape, and mask as the workspace sweep. FA2
used 44.7888 seconds for data generation and 1.3805 seconds for HBM residency;
FA4 used 44.6514 and 1.0693 seconds respectively. These preparation phases are
reported separately and excluded from execution time.

### Low-workspace extension protocol: 2026-08-24

The low-workspace extension measured 256, 384, 512, 640, 768, and 896MiB. It
kept the shape, seed, dtype, mask, host-output mode, two K/V buffers, one output
buffer, 32 preparation workers, 4,096-token CPU generation chunks, 100ms
PID-level NVML sampling, CPU affinity, and automatic Triton kernel selection
aligned with the original sweep.

Unlike the original scan, every workspace value ran in a separate Python
process, and the six processes ran serially on physical GPU3. Consequently,
Q/K/V and the pinned output were regenerated for every point. Preparation took
26.111-37.087 seconds per process and is excluded from the execution values.
The measured attention interval includes the final 7GiB D2H transfer into the
pinned host output.

The extension fixes `kv_chunk_tokens=2048` for all six points. The original
8,192-token chunk needs about 487.1MiB before even one query block fits, and a
4,096-token chunk still needs about 263.1MiB. A 2,048-token chunk was selected
as the largest conventional power-of-two setting that covers 256MiB while
leaving a useful 1,920-token resident query chunk at that boundary. Because K/V
chunk size changed from 8,192 to 2,048, the extension and original sweep are
not one directly comparable workspace curve.

GPU3 had no compute PID before the extension started or before any subsequent
point launched. An unrelated process began using physical GPU2 after the first
point started. It did not share GPU3 or its fixed CPU affinity, but the host was
therefore not globally exclusive during the complete extension.

## Results

### Original 1-14GiB sweep: 8,192-token K/V chunks

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

### Low-workspace extension: 2,048-token K/V chunks

| Workspace | Planned workspace | PID GPU peak | Resident Q | Q passes | Logical H2D | Execution | Tokens/s | Effective TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 MiB | 249.8 MiB | 792 MiB | 1,920 | 274 | 3,843 GiB | 114.839 s | 4,565 | 68.6 |
| 384 MiB | 383.9 MiB | 922 MiB | 4,352 | 121 | 1,701 GiB | 52.594 s | 9,969 | 149.9 |
| 512 MiB | 510.8 MiB | 1,066 MiB | 6,656 | 79 | 1,113 GiB | 40.844 s | 12,836 | 193.0 |
| 640 MiB | 637.8 MiB | 1,194 MiB | 8,960 | 59 | 833 GiB | 40.124 s | 13,067 | 196.4 |
| 768 MiB | 764.8 MiB | 1,316 MiB | 11,264 | 47 | 665 GiB | 39.106 s | 13,407 | 201.5 |
| 896 MiB | 891.8 MiB | 1,444 MiB | 13,568 | 39 | 553 GiB | 38.943 s | 13,463 | 202.4 |

All rows used automatic `128x64`, 8-warp, 3-stage Triton kernels, transferred
7.0GiB D2H, and completed with `status=success` and no failure message.

| Workspace | Torch allocated peak | Torch reserved peak | Process RSS peak | Output signature |
|---:|---:|---:|---:|---|
| 256 MiB | 217.8 MiB | 224.0 MiB | 32.630 GiB | identical |
| 384 MiB | 353.9 MiB | 354.0 MiB | 32.679 GiB | identical |
| 512 MiB | 480.8 MiB | 498.0 MiB | 32.677 GiB | identical |
| 640 MiB | 606.8 MiB | 626.0 MiB | 32.674 GiB | identical |
| 768 MiB | 732.8 MiB | 748.0 MiB | 32.680 GiB | identical |
| 896 MiB | 861.8 MiB | 876.0 MiB | 32.680 GiB | identical |

## GPU-resident FlashAttention 2 and 4

The FA2 run used `flash-attn 2.7.4.post1+nv26.1.42222806`, imported as
`flash_attn.flash_attn_func`. The FA4 run used `flash-attn-4 4.0.0b26`, imported
as `flash_attn.cute.flash_attn_func`, with CuTeDSL `4.6.0.dev0` and
`quack-kernels 0.5.3`. Both used PyTorch `2.10.0+cu128` on the same RTX 5090
(`sm_120`). Complete Q/K/V occupies 21GiB and the HBM output adds 7GiB.

| Backend | Q/K/V -> output | PID GPU peak | Torch peak | Execution | Tokens/s | Effective TFLOPS | vs FA4 |
|---|---|---:|---:|---:|---:|---:|---:|
| FlashAttention 2 | HBM -> HBM | 28.600 GiB | 28.109 GiB | 36.975 s | 14,180 | 213.2 | 1.020x |
| FlashAttention 4 | HBM -> HBM | 28.486 GiB | 28.000 GiB | 36.267 s | 14,456 | 217.3 | 1.000x |
| seqattn, 2GiB | Pinned DRAM -> pinned DRAM | 2.533 GiB | 1.969 GiB | 36.247 s | 14,464 | 217.4 | 0.999x |
| seqattn, 14GiB | Pinned DRAM -> pinned DRAM | 14.525 GiB | 13.971 GiB | 36.010 s | 14,559 | 218.9 | 0.993x |

At 2GiB, seqattn reduces PID-level GPU memory by 91.1% relative to FA4 while
the observed execution times differ by 20.8 milliseconds, or 0.06%. The 14GiB
seqattn row is 0.71% lower than FA4. Each row is one observation, so these
sub-percent differences should be treated as the same performance band rather
than a stable cross-run ranking.

The residency contracts differ. FA2 and FA4 start with Q/K/V in HBM and leave
output in HBM. The measured seqattn path starts with caller-owned pinned DRAM
and includes the final 7GiB D2H output transfer. A GPU consumer would need
seqattn's device-consumer path or another H2D transfer; that application-level
choice is not represented by the host-output rows above.

FA2 and FA4 produced identical values at all 40 sampled BF16 positions. The
14GiB seqattn row compared with either resident backend has:

```text
relative L2:  0.002958
max absolute: 3.052e-5
cosine:       0.99999586
```

These are sparse signature metrics, not full-output error norms. The different
BF16 values reflect different floating-point reduction orders; the repository's
reference-based tests remain the correctness authority.

### Why seqattn appeared faster than FA2

The result is not caused by FA2 falling back to an older GPU architecture.
`cuobjdump` confirmed that the installed FA2 extension contains native
`sm_120` cubins. A short 16K-token Nsight Systems capture, used only to inspect
kernel selection, recorded one kernel per resident backend:

| Backend | Captured kernel | 16K kernel time |
|---|---|---:|
| FA2 | `flash_fwd_kernel`, head dim 128, `128x64`, 4 warps | 34.652 ms |
| FA4 | `flash_fwd_sm120` CuTeDSL kernel | 34.134 ms |

FA4 is 1.5% faster in this short capture and 1.9% faster at 524K, so the newer
SM120-specific schedule explains most of the apparent FA2 gap. Seqattn uses a
shape-specific `128x64`, 8-warp, 3-stage Triton configuration, 8K K/V tiles,
and resident-Q chunks. At 14GiB it executes 3 Q passes and 192 K/V tile kernels;
at 2GiB it executes 19 Q passes and 1,216 K/V tile kernels. Kernel launch cost
is negligible relative to a 36-second attention call. The runtime pipelines
H2D and D2H with compute, and the flat 2-14GiB curve indicates that most copy
work is hidden for this shape.

The additional warps and deeper software pipeline are plausible reasons for
slightly better scheduling on this exact BF16, head-dimension-128,
single-segment shape. That is an inference from the selected configurations
and timings, not a register/occupancy conclusion from Nsight Compute counters.
The defensible claim is that current seqattn matches FA4 for this shape while
using much less HBM, not that it generally outperforms FlashAttention.

## Interpretation

### Low-workspace behavior

At 256MiB, the planner can retain only 1,920 query tokens, so the operator
makes 274 Q passes and logically transfers 3,843GiB H2D. Increasing the budget
to 384MiB cuts the pass count to 121 and execution time by 54.2%. Moving from
384MiB to 512MiB cuts execution time by another 22.3%.

The curve then begins to flatten. The 512MiB point executes in 40.844 seconds,
and increasing workspace to 896MiB improves that by a further 4.65% to 38.943
seconds. Across the full extension, 896MiB is 2.95x faster than 256MiB while
using 652MiB more PID-level GPU memory. Logical H2D falls by 6.95x, closely
tracking the reduction from 274 to 39 complete Q passes.

The 896MiB extension point is 1.94% slower than the original 1GiB observation,
but that difference is not an isolated workspace effect: the extension uses
2,048-token K/V chunks and independent process startup, while the original
point uses 8,192-token chunks within a one-process sweep. The defensible
cross-protocol conclusion is only that performance is already near the
original 1GiB range by 768-896MiB.

### Original 1-14GiB behavior

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

The six low-workspace processes also produced one identical sampled signature,
and it matches the signature from the original eight-point sweep. This checks
consistency across both planner choices and the 8,192-to-2,048 K/V chunk change.

## Limits and artifacts

- Each workspace value has one observation.
- Both workspace protocols used physical GPU3 only. The original sweep used one
  process, while the low-workspace extension used six serial independent
  processes; no multi-GPU throughput claim is implied.
- CPU affinity was fixed, but the complete host was not reserved exclusively.
- Data preparation is reported separately and excluded from execution time.
- The test exercises caller-owned DRAM streaming, not the paged or NVMe path.
- The FA2, FA4, and seqattn rows were separate processes and separate single
  observations; no confidence interval is available.
- Each low-workspace point is also a separate single observation. GPU3 was
  exclusive, but an unrelated workload occupied GPU2 during part of the run.
- The 16K Nsight captures identify kernel selection only and are not used as
  the primary 524K performance result.

The source-of-truth JSON is retained locally at:

```text
workspace/benchmarks/results/rtx5090_dram_workspace_524k_optimized_20260819/gpu3_auto_blackwell_single.json
workspace/benchmarks/results/rtx5090_flash_backend_524k_20260819/fa2_524288_gpu3.json
workspace/benchmarks/results/rtx5090_flash_backend_524k_20260819/fa4_524288_gpu3.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0250g.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0375g.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0500g.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0625g.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0750g.json
workspace/benchmarks/results/rtx5090_dram_workspace_524k_low_20260824/workspace_0875g.json
```
