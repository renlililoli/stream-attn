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
```

The plot and observation summary are regenerated with
`benchmarks/analyze_host_memory_roofline.py`.
