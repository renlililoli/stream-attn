# RTX 5090 Dynamic Multi-GPU Q Scheduling at 524K Tokens

## Summary

On August 26, 2026, current commit `7a3361888930681648cf7d1cf0d1d4f1ef44a475`
was benchmarked with two and three idle RTX 5090 GPUs. Physical GPU 0 remained
occupied by another process and was not used.

The primary comparison uses a fair tuned static schedule. Static Q sizes and
planner rates were taken from the corresponding converged dynamic run instead
of retaining the isolated-device starting value of 5,760 tokens.

| Implementation | GPUs | Selected samples (s) | Median (s) | Tokens/s | Effective TFLOPS | Speedup vs single SeqAttn 14 GiB | Speedup vs resident FA4 |
|---|---:|---|---:|---:|---:|---:|---:|
| SeqAttn, 14 GiB, historical | 1 | `36.010` | 36.010 | 14,559 | 218.9 | 1.00x | 1.01x |
| Resident FA4, historical | 1 | `36.267` | 36.267 | 14,456 | 217.3 | 0.99x | 1.00x |
| Tuned static | 2 | `18.295, 18.323, 18.343` | **18.323** | 28,614 | 430.1 | **1.97x** | **1.98x** |
| Dynamic, converged | 2 | `18.419, 18.431, 18.443` | **18.431** | 28,446 | 427.6 | **1.95x** | **1.97x** |
| Tuned static | 3 | `12.403, 12.431, 12.453` | **12.431** | 42,174 | 634.0 | **2.90x** | **2.92x** |
| Dynamic, converged | 3 | `12.507, 12.558, 12.587` | **12.558** | 41,748 | 627.6 | **2.87x** | **2.89x** |

After Q tuning, dynamic reaches 99.41% of tuned-static throughput on two GPUs
and 98.99% on three GPUs. Tuned static is 0.59% faster on two GPUs and 1.02%
faster on three GPUs. Dynamic therefore does not improve the steady result on
these idle, nominally homogeneous devices; its value is adaptation to changing
or heterogeneous device and host-link performance.

Three GPUs provide 1.47x the throughput of two GPUs for both scheduling modes.
Relative to the previous single-GPU rows, the current three-GPU host-output
path reaches approximately 2.9x speedup.

## Configuration

All current multi-GPU rows use:

```text
tokens:                 524,288
segments:               1
Q heads / KV heads:     56 / 56
head dimension:         128
dtype:                  BF16
causal:                 false
backend:                Triton
kernel profile:         block_m=128, block_n=64, warps=8, stages=3
KV chunk:               8,192
KV buffers:             2
output buffers:         1
output:                 pinned CPU tensor
dynamic initial Q:      5,760
dynamic minimum Q:      2,048
dynamic Q capacity:     23,040
```

The dynamic capacity is preallocated workspace, not the active Q size. No
steady-state workspace growth occurs. The controller stopped near 13K Q on
both the two- and three-GPU runs, well below the 23,040-token capacity.

The container image was
`diffsynth:cu128-roofline-fa4-20260824`, image digest
`sha256:0a5009b416016c52edbe8c05eedc9025e57ef47fde2c8f5e70775ba4a14cf1c5`.
It provided Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8, and Triton 3.7.1.

Two-GPU runs used physical GPUs 1 and 3:

| Container device | Physical GPU | UUID | PCI address |
|---|---:|---|---|
| `cuda:0` | 1 | `GPU-28e1c1eb-f738-21b5-909c-b025a2281165` | `81:00.0` |
| `cuda:1` | 3 | `GPU-35d69b8c-779a-bb45-263c-27587354c413` | `e1:00.0` |

CPU affinity was `160-191,224-255,416-447,480-511`. Host allocations used
`numactl --interleave=5,7` with Docker `--cpuset-mems=5,7`.

Three-GPU runs added physical GPU 2:

| Container device | Physical GPU | UUID | PCI address |
|---|---:|---|---|
| `cuda:0` | 1 | `GPU-28e1c1eb-f738-21b5-909c-b025a2281165` | `81:00.0` |
| `cuda:1` | 2 | `GPU-a9f0c52d-3d52-fd1a-71b9-44f2a59366f7` | `a1:00.0` |
| `cuda:2` | 3 | `GPU-35d69b8c-779a-bb45-263c-27587354c413` | `e1:00.0` |

CPU affinity was `160-255,416-511`. Physical GPU 2's NUMA node has no local
memory, so host allocations remained interleaved across memory-bearing nodes 5
and 7.

## Q Adaptation and PCIe Contention

The isolated RTX 5090 Q knee of approximately 5,760 tokens is not the correct
steady Q for these concurrent runs. Dynamic scheduling observed substantially
lower per-device H2D rates while multiple GPUs streamed full K/V segments.

| GPUs | Device | Final Q | Q tokens completed | Tasks | Work share | Effective TFLOPS EMA | Observed H2D GB/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | physical 1 | 12,288 | 264,864 | 22 | 50.52% | 218.09 | 18.70 |
| 2 | physical 3 | 13,056 | 259,424 | 20 | 49.48% | 215.95 | 17.22 |
| 3 | physical 1 | 12,928 | 174,976 | 14 | 33.37% | 216.42 | 17.82 |
| 3 | physical 2 | 12,928 | 179,584 | 14 | 34.25% | 216.75 | 17.45 |
| 3 | physical 3 | 13,056 | 169,728 | 13 | 32.37% | 216.09 | 17.06 |

These concurrent H2D values differ materially from the approximately
56.7 GB/s isolated interleaved calibration. This supports treating per-device
bandwidth as an online measurement under the actual multi-GPU traffic pattern,
not as a reusable isolated constant.

Allowing Q to grow also reduced repeated full-K/V scans during controller
convergence. Two-GPU dynamic improved from 20.280 seconds on its first call to
an 18.431-second steady median. Three-GPU dynamic improved from 15.910 seconds
to a 12.558-second steady median. This is a convergence comparison within
dynamic mode, not a static-versus-dynamic speedup claim.

## Fair Static Schedules

The final static schedules use the dynamic run's final Q sizes and measured
performance estimates:

| GPUs | Physical GPU | Static Q | Assigned global Q range | Assigned tokens | Tasks |
|---:|---:|---:|---|---:|---:|
| 2 | 1 | 12,288 | `[0, 263168)` | 263,168 | 22 |
| 2 | 3 | 13,056 | `[263168, 524288)` | 261,120 | 20 |
| 3 | 1 | 12,928 | `[0, 177152)` | 177,152 | 14 |
| 3 | 2 | 12,928 | `[177152, 354560)` | 177,408 | 14 |
| 3 | 3 | 13,056 | `[354560, 524288)` | 169,728 | 13 |

An intermediate two-GPU static run using the earlier dynamic values
`9344/10368` measured 18.503 seconds. Updating static to `12288/13056` reduced
that to 18.323 seconds. An earlier terminal-only `5760/5760` control measured
20.302 seconds, but it is not used in the primary comparison because it lacks
the committed raw artifact and uses an unfair isolated-device Q.

## Baseline Provenance and Timing Boundaries

The current multi-GPU measurements time one complete runner invocation around
the CPU-backed Q/K/V inputs, including streamed H2D, attention, and final D2H
into a reusable pinned CPU output. Each static process used one warmup and
three measured calls. Dynamic recorded five calls and the table selects the
last three after Q convergence.

The single-GPU rows are existing August 19, 2026 artifacts from an older
revision and are included because no new single-GPU rerun was requested:

- SeqAttn 14 GiB: 36.010 seconds, Q=251,392, host output.
- SeqAttn 2 GiB: 36.247 seconds, Q=28,416, host output.
- Resident FA4: 36.267 seconds, GPU-resident output.

The resident FA4 row does not include a final GPU-to-CPU output copy. Its
timing boundary is therefore more favorable than the current multi-GPU
host-output rows. The single-GPU numbers also use a different code revision,
process, and workspace regime, so their speedups are historical context rather
than a same-process controlled comparison.

All current two- and three-GPU static and dynamic runs produced finite outputs
and identical sampled BF16 output signatures at five sequence positions.

## Artifacts

The benchmark entry point is `benchmarks/multigpu_524k.py`. Raw current-run
artifacts are committed under
`docs/experiments/rtx5090_dynamic_multigpu_524k_20260826/`:

```text
two_gpu_dynamic.json
two_gpu_static_tuned.json
two_gpu_static_tuned_final.json
three_gpu_dynamic.json
three_gpu_static_tuned.json
```

Historical single-GPU source artifacts remain outside this repository at:

```text
workspace/benchmarks/results/rtx5090_dram_workspace_524k_optimized_20260819/gpu3_auto_blackwell_single.json
workspace/benchmarks/results/rtx5090_flash_backend_524k_20260819/fa4_524288_gpu3.json
```
