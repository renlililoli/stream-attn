# Host-memory roofline validation: A30 and RTX 5090

Date: 2026-08-24 to 2026-08-26.

This report consolidates the current checked-in A30 and RTX 5090 calibration
results. It replaces the separate experiment plan, intermediate roofline
reports, memory-population note, and workspace-sweep narratives that remain
available through Git history.

The calibration method itself remains in
[`q_chunk_calibration.md`](q_chunk_calibration.md).

## Model

For BF16 or FP16 MHA with measured concurrent pinned-host bandwidth `B` in
decimal GB/s and measured resident attention throughput `P` in TFLOP/s:

```text
P_stream(q_effective) = min(B * q_effective / 1000, P)
q_star                = 1000 * P / B
q_effective           = sequence_tokens / ceil(sequence_tokens / q_requested)
```

The experiments froze `B` and `P` before measuring the Q sweep. K/V chunk size,
shape, backend, GPU, CPU affinity, and memory policy were fixed within each
series.

## NVIDIA A30

<p align="center">
  <img src="assets/latest-a30-host-memory-roofline.svg" alt="A30 final host-memory roofline validation" width="100%">
</p>

| Setting | Value |
|---|---:|
| GPU and backend | NVIDIA A30, SM80, FlashAttention 2 |
| Shape | 409,600 tokens, BF16 MHA, 56 heads, head dimension 128 |
| K/V chunk | 4,096 tokens |
| Concurrent H2D roof | 12.3577 GB/s |
| Resident FA2 roof | 92.1634 TFLOP/s |
| Streaming plateau | 87.3816 TFLOP/s |
| Predicted knee | 7,457.99 effective Q tokens |
| Inferred knee | 7,307.76 effective Q tokens |
| Difference | -2.01% |

The low-Q branch followed the independently frozen bandwidth prediction within
about 2%. Above the knee, throughput settled near 87-88 TFLOP/s, approximately
94.8% of the complete resident FA2 roof.

Each Q point used one warmup and three measured executions in one randomized
process. This validates the first-order knee but does not quantify
independent-process or thermal variance.

Raw evidence:

- [`experiments/a30_host_memory_roofline_experiment0_20260824/prediction.json`](experiments/a30_host_memory_roofline_experiment0_20260824/prediction.json)
- [`experiments/a30_host_memory_roofline_experiment0_20260824/observations.json`](experiments/a30_host_memory_roofline_experiment0_20260824/observations.json)

## NVIDIA RTX 5090

<p align="center">
  <img src="assets/latest-rtx5090-host-memory-roofline.svg" alt="RTX 5090 host-memory roofline validation for two memory policies" width="100%">
</p>

The RTX 5090 experiment fixed the resident FA4 roof and changed only pinned
host-memory placement. The two policies produced different measured bandwidth
and therefore different predicted Q knees.

| Setting | Value |
|---|---:|
| GPU and resident backend | NVIDIA RTX 5090, SM120, FlashAttention 4 |
| Streaming backend | SeqAttn built-in Triton |
| Shape | 524,288 tokens, BF16 MHA, 56 heads, head dimension 128 |
| K/V chunk | 4,096 tokens |
| Resident FA4 roof | 213.3230 TFLOP/s |

| Host-memory policy | Concurrent H2D | Predicted knee | Inferred knee | Difference | Plateau |
|---|---:|---:|---:|---:|---:|
| `membind=5` | 37.2840 GB/s | 5,721.56 | 5,748.77 | +0.48% | 203.411 TFLOP/s |
| `interleave=5,7` | 56.7170 GB/s | 3,761.18 | 3,754.30 | -0.18% | 203.072 TFLOP/s |

At requested `q=4096`, interleaving increased throughput from 150.02 to
203.07 TFLOP/s, a 35.4% gain. The result demonstrates that CPU memory
population and NUMA placement are part of the attention configuration, not
incidental machine metadata.

Balanced final reruns used `q=5888,6272,6784` for `membind=5` and
`q=3840,4096,4480` for `interleave=5,7`. Each point ran in a fresh process with
one warmup and three measured executions. Contaminated runs were excluded from
the final observations.

Raw evidence:

- [`experiments/rtx5090_host_memory_roofline_experiment0_20260824/prediction.json`](experiments/rtx5090_host_memory_roofline_experiment0_20260824/prediction.json)
- [`experiments/rtx5090_host_memory_roofline_experiment0_20260824/observations.json`](experiments/rtx5090_host_memory_roofline_experiment0_20260824/observations.json)
- [`experiments/rtx5090_host_memory_roofline_experiment0_20260824/pcie_memory_population_diagnostic.json`](experiments/rtx5090_host_memory_roofline_experiment0_20260824/pcie_memory_population_diagnostic.json)
- [`experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/prediction.json`](experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/prediction.json)
- [`experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/comparison_observations.json`](experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/comparison_observations.json)

## Conclusions

1. Measured concurrent H2D bandwidth and measured resident attention
   throughput predict the useful Q-chunk region on both systems.
2. A GPU model name alone is insufficient. Host memory population, NUMA
   policy, backend, shape, and K/V tile all affect the result.
3. The predicted knee is a sweep center, not an automatic production default.
   Validate aligned candidates around it and retest the winner independently.
4. These results do not transfer to a different topology without
   recalibration.

The README figures are regenerated from the checked-in observations with
`benchmarks/plot_latest_readme_results.py`.
