# RTX 5090 Host-Memory Roofline Experiment 0

Date: 2026-08-24

Status: in progress. This report contains the independent calibration and the
first four prospectively measured Q points. The remaining fine/coarse sweep,
independent process replications, and profiler captures are still running or
pending.

## Question

For exact non-causal BF16 MHA with a fixed 4,096-token K/V chunk, does SeqAttn
throughput follow the independently calibrated host-memory roofline

```text
P(q) = min(B_P * q_effective, P_FA4)?
```

The workspace is held at 4GiB and only resident Q is varied. The prediction was
committed before any new Q-sweep result was launched.

<p align="center">
  <img src="assets/rtx5090-host-memory-roofline-experiment0.svg" alt="RTX 5090 host-memory roofline prediction and first measured Q points" width="100%">
</p>

## Locked Shape

```text
GPU:             NVIDIA GeForce RTX 5090, physical GPU3
tokens:          524,288
Q/K/V heads:     56 / 56 / 56
head dimension:  128
dtype:           BF16
causal:          false
KV chunk:        4,096
kernel:          BLOCK_M=128, BLOCK_N=64, warps=8, stages=3
KV buffers:      2
output:          pinned host DRAM
```

Input tensors are generated deterministically in pinned DRAM with 32 CPU
workers and 4,096-token chunks. Data preparation is outside attention timing.

## Independent Inputs

The two model inputs have different sources and must not be conflated.

| Input | Measurement | Frozen value |
|---|---|---:|
| `B_P` | Median concurrent H2D service rate while the SeqAttn update kernel is active; equal-weight median of `q_compute=8192` and `16384` | 37.2840 GB/s |
| `P_FA4` | Median of 10 CUDA-event FA4 throughput samples at the complete 524K resident shape, after 5 warmups | 213.3230 TFLOP/s |

The resulting prospective prediction is:

```text
q_star_predicted = 5,721.56
q_95_predicted   = 5,435.49
q_star_aligned   = 5,760
workspace(q=5760, k=4096) = 573.4609375 MiB
```

The pre-registration is
[`prediction.json`](experiments/rtx5090_host_memory_roofline_experiment0_20260824/prediction.json).

## First Observations

Each row is one independent process with one warmup and five measured
executions. The primary throughput uses a CUDA-event interval ending after the
last finalize on the compute stream, so it excludes only the final D2H tail.
The full host-output wall interval is retained separately.

| Requested q | Effective q | Q passes | Predicted TFLOPS | Measured pipeline TFLOPS | Measured / predicted | Measured / FA4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 4,096.0 | 128 | 152.72 | 150.02 | 98.24% | 70.33% |
| 5,504 | 5,461.3 | 96 | 203.62 | 198.86 | 97.66% | 93.22% |
| 5,760 | 5,698.8 | 92 | 212.47 | 204.93 | 96.45% | 96.07% |
| 8,192 | 8,192.0 | 64 | 213.32 | 206.21 | 96.66% | 96.66% |

The first four observations are within 1.76% to 3.55% of the prospective
model. In particular:

1. At `q=4096`, the model predicts the PCIe branch rather than the compute
   roof. Measured throughput is 98.24% of that prediction.
2. At `q=5504`, the exact finite-N effective Q is 5,461.3 tokens. Measured
   throughput is 97.66% of the predicted PCIe roof at that point.
3. At the aligned intersection `q=5760`, SeqAttn reaches 96.07% of the
   independently measured resident FA4 roof.
4. Increasing Q to 8,192 raises throughput only from 204.93 to 206.21 TFLOP/s,
   which is early evidence of the predicted plateau.

The observed plateau is currently about 3.3% below `P_FA4`. This is not a
failure of the host-link intensity model: FA4 and SeqAttn are different
kernels. It measures the cross-kernel efficiency factor that the experiment
was designed to expose:

```text
eta = P_SeqAttn,plateau / P_FA4 ~= 0.967
```

The full sweep is needed before freezing `eta` or the observed 5% knee.

## Timing Boundary

Pipeline and full wall throughput differ by less than 0.02% at these long
sequence points. The final 7GiB output transfer is almost entirely overlapped
with preceding Q chunks; only its final tail lies outside the compute-stream
completion event. CUDA-event and host wall timers are different clock domains,
so sub-millisecond signed differences must not be interpreted as standalone
copy latency.

## Why H2D Is 37GB/s Instead of 60GB/s

The PCIe endpoint and root port both negotiate PCIe 5.0 x16. After 128b/130b
encoding, the link carries at most 63.015GB/s before transaction-layer and
implementation overhead, so protocol overhead alone cannot explain a stable
37.3GB/s result.

The host-memory topology does explain it. This two-socket EPYC 9754 system has
256GiB total DRAM. Linux EDAC reports memory only on channel indices 3 and 9 of
each socket, with 64GiB on each populated channel. With NPS4 enabled, only NUMA
nodes 1, 3, 5, and 7 have memory; nodes 0, 2, 4, and 6 have zero local memory.
GPU3 is attached to node5, so the locked NUMA-local allocation is supplied by
one populated DDR5 channel.

<p align="center">
  <img src="assets/rtx5090-pcie-memory-population-diagnostic.svg" alt="H2D bandwidth rises from 37GB/s to 56.76GB/s when pinned pages are interleaved across two populated memory nodes" width="100%">
</p>

The causal check used idle physical GPU2 and the exact runtime payload: two
back-to-back 56MiB BF16 copies, 10 warmups, and 50 measured samples.

| Pinned-memory policy | Median GB/s | p10 GB/s | p90 GB/s | Interpretation |
|---|---:|---:|---:|---|
| `membind=5` | 36.145 | 35.432 | 36.595 | one populated channel |
| `membind=7` | 36.923 | 36.399 | 37.236 | one populated channel |
| `interleave=5,7` | 56.760 | 56.756 | 56.764 | two populated channels |

Interleaving the same pinned allocation across the socket's two populated
memory nodes raises H2D bandwidth by 53.7% and reaches 90.1% of the encoded
PCIe 5.0 x16 line rate. This rules out an intrinsic 37GB/s RTX 5090 copy-engine
ceiling. The original 37.284GB/s calibration is a valid measured input for the
locked single-node experiment, but its correct name is the current machine's
effective host-supply roof, not the theoretical PCIe roof.

This distinction matters for portability. Fully populating more memory
channels, or deliberately interleaving pinned pages across channels, increases
`B_P`, moves the predicted knee to smaller resident Q, and changes the hardware
profile without changing the SeqAttn kernel.

One GPU3 recheck performed after an unrelated training process started at
2026-08-24 06:28:45 UTC fell to 27.50GB/s and was rejected as contaminated.
The `q=4736` run also completed after that start time and is excluded from
formal observations. The q-sweep remains paused until an uncontended target GPU
is available.

## Preliminary Bandwidth-Shift Validation

Experiment 0B changes only pinned-memory placement from `membind=5` to
`interleave=5,7`. The shape, GPU, K/V chunk, kernel profile, buffer counts,
workspace guardrail, output mode, and FA4 roof remain fixed.

The GPU3 calibration was completed before the new q-sweep:

| Measurement | Median GB/s |
|---|---:|
| Single-node concurrent `B_P` | 37.2840 |
| Interleaved bare H2D | 56.7552 |
| Interleaved concurrent, `q_compute=8192` | 56.7196 |
| Interleaved concurrent, `q_compute=16384` | 56.7144 |
| Frozen interleaved `B_P` | **56.7170** |

This bandwidth increase changes the prospective prediction from:

```text
single node: q_star=5721.56, q_95=5435.49
interleaved: q_star=3761.18, q_95=3573.12
```

The predicted full-roof intersection moves left by 34.3%. The prediction was
committed before any interleaved q result at
`docs/experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/prediction.json`.

<p align="center">
  <img src="assets/rtx5090-host-memory-roofline-bandwidth-shift.svg" alt="Single-node and interleaved host-memory rooflines with measured SeqAttn points" width="100%">
</p>

The first uncontended quick points use one warmup and three measured repeats:

| Policy | Requested q | Effective q | Predicted TFLOPS | Pipeline TFLOPS | Observed / predicted |
|---|---:|---:|---:|---:|---:|
| Single node | 4,096 | 4,096.0 | 152.72 | 150.02 | 98.24% |
| Interleaved | 3,584 | 3,566.6 | 202.29 | 196.98 | 97.38% |
| Interleaved | 3,712 | 3,692.2 | 209.41 | 200.32 | 95.66% |
| Interleaved | 4,096 | 4,096.0 | 213.32 | 203.35 | 95.32% |

At identical `q=4096`, the measured throughput rises from 150.02 to
203.35TFLOP/s, a 35.5% gain caused only by the host-memory placement change.
The new transition is already visible around 3.6K-3.8K effective Q, close to
the independently predicted `q_star=3761.18`.

One `q=3840` process produced per-repeat pipeline times of 38.33, 62.83, and
73.90 seconds. The progressive slowdown coincided with unrelated host jobs
consuming tens of GiB and reducing node7 free memory. A requested rerun then
stalled before GPU execution while building the approximately 28GiB pinned
Q/K/V/output allocation; node7 had only about 8.6GiB free, below its roughly
14GiB interleaved share. These runs are retained but excluded from the quick
knee plot. GPU `Exclusive_Process` prevents CUDA-context contention but does
not isolate host DRAM capacity or bandwidth.

This comparison is preliminary because only three clean interleaved points are
available. It nevertheless supports the direction and approximate magnitude
of the predicted knee shift. Remaining points and the `q=3840` rerun require
an uncontended host-memory window.

The experiment was paused on 2026-08-24 with no benchmark process left
running. A second `q=3840` attempt was stopped after more than five minutes of
host-side preparation: its RSS had reached approximately 24GiB, its main
thread occupied one CPU core, GPU utilization remained 0%, and it had not
written a result. A benchmark-only parallel pinned-allocation path has been
implemented but not yet validated. Compilation passed; its first smoke test
was blocked when an unrelated process acquired physical GPU3 under
`Exclusive_Process`. On resume, validate that path with a small allocation and
streaming run before launching another full 524K-token point.

## Current Conclusion

The prospective model has passed its first out-of-sample check. The low-Q
point follows the independently measured H2D slope, and the predicted
intersection lands in the measured transition to a roughly 206TFLOP/s
plateau. The evidence is already stronger than the old retrospective
20.9GB/s calculation because neither `B_P` nor `P_FA4` was derived from these Q
points.

This is still an interim result. Publication claims require:

- completion of the preregistered fine and coarse Q sets;
- three independent processes per Q;
- an observed-knee definition based on the measured high-Q plateau;
- Nsight Systems traces below, near, and above the knee;
- the final fixed-4K workspace validation.

## Artifacts

```text
Calibration:
workspace/benchmarks/results/rtx5090_host_memory_roofline_experiment0_20260824/calibration/

Sentinel Q results:
workspace/benchmarks/results/rtx5090_host_memory_roofline_experiment0_20260824/q_sweep_k4096_sentinel/

Fine sweep, round 1:
workspace/benchmarks/results/rtx5090_host_memory_roofline_experiment0_20260824/q_sweep_k4096_fine_round1/

Machine-readable report data:
docs/experiments/rtx5090_host_memory_roofline_experiment0_20260824/observations.json

PCIe/memory-population diagnostic summary:
docs/experiments/rtx5090_host_memory_roofline_experiment0_20260824/pcie_memory_population_diagnostic.json

Bandwidth-shift comparison summary:
docs/experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/comparison_observations.json

Raw PCIe/memory-population runs:
workspace/benchmarks/results/rtx5090_pcie_memory_population_20260824/

Raw interleaved calibration and q results:
workspace/benchmarks/results/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/
```

The original plot and observation summary are regenerated with
`benchmarks/analyze_host_memory_roofline.py`. The bandwidth-shift comparison is
regenerated with `benchmarks/analyze_host_memory_roofline_comparison.py`.
