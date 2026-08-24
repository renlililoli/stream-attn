# RTX 5090 Host-Memory Roofline Experiment 0 TODO

Date: 2026-08-24

## Objective

Validate the host-memory roofline with one controlled independent variable:
fix the K/V chunk at 4,096 tokens and sweep only the resident Q chunk.

The experiment must independently measure the two model inputs before any new
Q-sweep result is inspected:

```text
B_P              concurrent pinned-host-to-GPU K/V bandwidth
P_FA4            resident FlashAttention 4 empirical compute roof
q_star_predicted P_FA4 / B_P for BF16 MHA
```

The Q sweep then tests whether SeqAttn reaches its measured plateau near the
pre-registered prediction. A later fixed-4K workspace sweep maps the validated
Q knee back to an HBM budget.

## Current Execution Status

```text
branch:              rtx5090-low-workspace-sweep-20260824
documentation commit: 3d13d75
calibration script:  benchmarks/host_memory_roofline_calibration.py
experiment container: seqattn-roofline-gpu3
committed image tag: diffsynth:cu128-roofline-fa4-20260824
committed image ID:  sha256:0a5009b416016c52edbe8c05eedc9025e57ef47fde2c8f5e70775ba4a14cf1c5
image created:       2026-08-24T03:49:56.173587513Z
physical GPU:        GPU3
container cpuset:    160-191,416-447
container mem nodes: 5
container security:  seccomp=unconfined for NUMA policy calls
```

The dedicated container is based on `diffsynth:cu128` and retains PyTorch
`2.10.0+cu128` and CUDA 12.8. It adds the fixed resident-roof stack used by the
2026-08-19 report:

```text
flash-attn-4:                4.0.0b26
nvidia-cutlass-dsl:          4.6.0.dev0
nvidia-cutlass-dsl-libs-base: 4.6.0.dev0
quack-kernels:               0.5.3
cuda-python:                 12.9.4
apache-tvm-ffi:              0.1.13.post3
torch-c-dlpack-ext:          0.1.5
```

Completed calibration artifacts are under:

```text
workspace/benchmarks/results/rtx5090_host_memory_roofline_experiment0_20260824/
```

The first formal 4K K/V H2D calibration used 10 warmups and 50 samples:

| Measurement | Median GB/s | p10 GB/s | p90 GB/s |
|---|---:|---:|---:|
| Bare, two 56MiB copies | 37.264 | 36.995 | 37.341 |
| Concurrent, `q_compute=8192` | 37.300 | 36.965 | 37.357 |
| Concurrent, `q_compute=16384` | 37.268 | 37.037 | 37.320 |

The two concurrent medians differ by 0.085%, so the initial scalar-bandwidth
stability check passes. The concurrent input is frozen as the equal-weight
median of the two per-compute-load medians:

```text
B_P = 37.2840346 GB/s
```

The apparent gap to PCIe 5.0 x16 has now been isolated to host-memory
population rather than link negotiation or the GPU copy engine:

```text
CPU:                     2 x AMD EPYC 9754
memory capacity:         256 GiB total, 128 GiB per socket
populated channel IDs:   3 and 9 on each socket
NPS4 memory nodes:       node1, node3, node5, node7
NPS4 CPU-only nodes:     node0, node2, node4, node6
GPU3 local memory node:  node5
```

On idle physical GPU2, using the exact two-copy 112MiB BF16 payload:

| Pinned-memory policy | Median GB/s | p10 GB/s | p90 GB/s |
|---|---:|---:|---:|
| `membind=5` | 36.145 | 35.432 | 36.595 |
| `membind=7` | 36.923 | 36.399 | 37.236 |
| `interleave=5,7` | 56.760 | 56.756 | 56.764 |

The interleaved result reaches 90.07% of the 63.015GB/s PCIe 5.0 x16 encoded
line rate. A single DDR5-4800 channel has a 38.4GB/s theoretical payload rate,
and the measured 36-37GB/s single-node ceiling is consistent with that supply
limit. Therefore retain 37.284GB/s as the measured model input for the locked
single-node experiment, but label it `host-supply-limited` rather than a
generic PCIe 5.0 ceiling.

Raw diagnostic artifacts are under:

```text
workspace/benchmarks/results/rtx5090_pcie_memory_population_20260824/
```

The first fine q-sweep is complete. The `q=4736` result completed while
physical GPU3 was shared by an unrelated training process and must not be used
as an uncontended formal observation. Replication of the original single-node
points is deferred until after the NUMA-interleaved knee-shift experiment.

NUMA sampling during the live pinned allocation reported about 740MiB on node
5 out of 815MiB total process memory. The remaining node 1/7 pages were mainly
shared library mappings; the process heap was on node 5.

FA4 import and compilation were first validated with a 16K-token smoke test.
The formal 524,288-token run then used 5 warmups and 10 measured CUDA-event
samples:

```text
median latency:  36.945390625 s
P_FA4 median:    213.3230037 TFLOP/s
P_FA4 p10/p90:   212.5419044 / 213.6852822 TFLOP/s
```

The independently frozen prospective prediction is:

```text
q_star_predicted:       5721.5644
q_95_predicted:         5435.4862
q_star_aligned_128:     5760
workspace at q=5760:    601317376 bytes = 573.4609375 MiB
```

The machine-readable pre-registration is committed at
`docs/experiments/rtx5090_host_memory_roofline_experiment0_20260824/prediction.json`.

The first prospective observations are now available:

| requested q | effective q | predicted TFLOPS | pipeline TFLOPS | observed / predicted |
|---:|---:|---:|---:|---:|
| 4096 | 4096.0 | 152.72 | 150.02 | 98.24% |
| 5504 | 5461.3 | 203.62 | 198.86 | 97.66% |
| 5760 | 5698.8 | 212.47 | 204.93 | 96.45% |
| 8192 | 8192.0 | 213.32 | 206.21 | 96.66% |

The live report is
`docs/rtx5090_host_memory_roofline_experiment0_2026-08-24.md`. The remaining
fine/coarse points and independent process replications are still pending.

## Locked Primary Configuration

```text
GPU:                NVIDIA GeForce RTX 5090, physical GPU3
GPU PCI BDF:        0000:E1:00.0
GPU NUMA node:      5, subject to topology re-verification
CPU affinity:       160-191,416-447
tokens:             524288
segments:           1
q_heads:            56
kv_heads:           56
head_dim:           128
dtype:              bfloat16
causal:             false
kv_chunk_tokens:    4096
block_m / block_n:  128 / 64
num_warps:          8
num_stages:         3
num_kv_buffers:     2
num_output_buffers: 1
output_mode:        host
```

Do not run primary measurements concurrently on GPU2 and GPU3. Other GPU jobs
can contend for host DRAM or PCIe resources even when they use a different
device. Record all compute PIDs before every formal run.

## Measurement Definitions

For a 4,096-token BF16 MHA K/V tile:

```text
K bytes:     58,720,256 bytes = 56 MiB
V bytes:     58,720,256 bytes = 56 MiB
K + V bytes: 117,440,512 bytes = 112 MiB
```

The transfer benchmark must issue two separate 56MiB copies, matching the
runtime. Report decimal bandwidth for model use:

```text
B_GBps = bytes / seconds / 1e9
```

Do not label MiB/s or GiB/s as GB/s.

For requested resident Q chunk `q`, define:

```text
q_passes    = ceil(N / q)
q_effective = N / q_passes
```

The finite-N prediction uses the exact staircase:

```text
P_predicted(q) = min(B_P * q_effective, P_FA4)
```

The continuous intersection remains:

```text
q_star_predicted = P_FA4 / B_P
```

For the 5% plateau definition, compare the observed threshold with:

```text
q_95_predicted = 0.95 * q_star_predicted
```

or estimate the observed intersection as `q_observed_5pct / 0.95`. Do not
compare the 95% threshold directly with the 100% roof intersection.

## Phase 0: Preserve Existing Evidence

- [x] Retain the original 1-14GiB `kv_chunk=8192` sweep as observed data.
- [x] Retain the 256-896MiB `kv_chunk=2048` extension as observed data.
- [x] Label the old 218.5TFLOP/s plateau as an empirical operator roof.
- [x] Do not use the old 20.9GB/s implied bandwidth as a predictive input.
- [ ] Keep retrospective consistency checks separate from prospective results.

## Phase 1: Topology and NUMA Verification

- [x] Save `nvidia-smi topo -m`.
- [x] Save `lscpu -e=CPU,NODE,SOCKET`.
- [x] Save `numactl --hardware`.
- [x] Save `/sys/bus/pci/devices/0000:E1:00.0/numa_node`.
- [x] Run formal jobs with CPU and memory binding to node 5.
- [x] Verify actual pinned-memory page placement with `numastat -p PID` and
      `/proc/PID/numa_maps` during a live allocation.
- [ ] Record driver, CUDA, PyTorch, Triton, FA4, committed image ID, clocks,
      temperature, and active compute PIDs.

NUMA STREAM is optional supporting evidence. Its result must not be substituted
for pinned H2D bandwidth in the roofline model.

## Phase 2: Bare Pinned H2D Calibration

- [x] Add a benchmark that allocates NUMA-local pinned host K and V tensors.
- [x] Allocate separate GPU K and V destinations matching runtime layout.
- [x] Issue two back-to-back asynchronous 56MiB copies on one H2D stream.
- [x] Place timing events after any buffer-free wait and around only the two
      copies.
- [x] Run at least 10 warmups and 50 measured samples.
- [x] Report raw bytes, each duration, median, p10, p90, GB/s, and GiB/s.
- [ ] Optionally measure 2K and 8K payloads as a fixed-latency diagnostic.
- [x] Explain the 37GB/s ceiling with a single-node versus two-node interleave
      experiment; the latter reaches 56.760GB/s on the same PCIe 5.0 x16 GPU.

Primary output:

```text
results/experiment0/h2d_bare.json
```

## Phase 3: Concurrent Pinned H2D Calibration

- [x] Reuse the exact SeqAttn update kernel and `128x64/8/3` launch profile.
- [x] Run representative GPU compute while the copy stream transfers the next
      two 56MiB K/V buffers.
- [x] Measure at least `q_compute=8192` and `q_compute=16384` without using any
      new Q-sweep result to select them.
- [x] Use preallocated timing events so event creation is outside measurement.
- [x] Report copy-service time after `kv_free`, excluding compute backpressure.
- [x] Compare the two compute-load medians. If they differ by more than 5%, do
      not freeze a single scalar `B_P` without documenting the dependence.
- [ ] Capture one Nsight Systems trace to verify copy/compute overlap and the
      placement of `kv_ready` and `kv_free` waits.

Primary output:

```text
results/experiment0/h2d_concurrent.json
```

## Phase 4: Resident FA4 Calibration

- [x] Use the exact 524,288-token BF16 MHA shape and non-causal mask.
- [x] Place Q/K/V in HBM before the timing interval.
- [x] Time only the FA4 call with CUDA events.
- [x] Exclude host-to-GPU preparation and GPU-to-host output copies.
- [x] Run at least 5 warmups and 10 measured repetitions.
- [x] Report all durations and median/p10/p90 effective TFLOP/s.
- [x] Record output signature and finite-value checks.
- [x] Validate the fixed FA4 stack with a 16K-token compile/execution smoke.

Primary output:

```text
results/experiment0/fa4_resident.json
```

## Phase 5: Freeze Prediction Before Q Sweep

- [x] Select and document the concurrent-bandwidth aggregation rule.
- [x] Compute `q_star_predicted = P_FA4 / B_P` using decimal SI units.
- [x] Compute `q_95_predicted = 0.95 * q_star_predicted`.
- [x] Align candidate Q chunks to `BLOCK_M=128` only after retaining the raw
      continuous predictions.
- [x] Use `estimate_workspace_bytes()` to predict the corresponding 4K HBM
      budget; do not duplicate the planner memory formula by hand.
- [x] Write and commit `prediction.json` before launching the new Q sweep.

Required artifact fields:

```text
p_fa4_tflops
b_concurrent_bytes_per_second
b_concurrent_gbps
q_star_predicted
q_95_predicted
q_star_aligned_128
workspace_star_predicted_bytes
calibration input paths and hashes
git commit
timestamp
prediction_created_before_q_sweep=true
```

## Phase 6: Fixed-4K Q Sweep

The workspace budget is a non-variable guardrail. A 4GiB budget is sufficient
for all planned Q chunks and must remain fixed.

Coarse Q values:

```text
2048 4096 6144 8192 10240 12288 16384 24576 32768
```

Fine values are selected only after `prediction.json` is frozen. Use 512-token
spacing around the prediction while keeping every value divisible by 128.

- [ ] Run each Q value with `kv_chunk=4096` and the locked kernel profile.
- [ ] Use at least one warmup and five measured repetitions per process.
- [ ] For publication results, collect three independent processes per Q value.
- [ ] Randomize Q-value execution order within each replication round.
- [ ] Ensure Triton initialization and steady-state specializations are warmed
      before primary timing.
- [ ] Record requested Q, aligned Q, effective Q, Q passes, execution samples,
      effective TFLOP/s, logical K/V H2D, one-time Q H2D, D2H, and signatures.
- [x] Add compute-stream completion timing so the primary roofline metric can
      exclude the final D2H tail. Keep full host-output wall time as a secondary
      application metric.

Primary output directory:

```text
results/experiment0/q_sweep_k4096/
```

## Phase 7: Analysis and Observed Knee

- [ ] Define the measured plateau from the predeclared high-Q points.
- [ ] Compute `q_observed_5pct` and the corrected intersection estimate
      `q_observed_5pct / 0.95`.
- [ ] Report a 1% threshold only as a secondary sensitivity result.
- [ ] Plot measured SeqAttn throughput, the exact staircase PCIe roof, and the
      FA4 horizontal roof.
- [ ] Plot normalized `P_SeqAttn/P_FA4` against
      `q_effective/q_star_predicted`.
- [ ] Report `eta = P_SeqAttn_plateau / P_FA4` to test whether FA4 is a valid
      cross-kernel empirical roof for this first experiment.
- [ ] Report prediction error using matching threshold definitions.
- [ ] Capture Nsight Systems traces below, near, and above the predicted knee.

## Phase 8: Fixed-4K Workspace Validation

Existing workspace scans cannot replace this phase because they use 2K or 8K
K/V chunks. Keep `kv_chunk=4096` fixed and let only the workspace budget control
resident Q.

Initial budgets:

```text
512 640 768 896 1024 1280 1536 MiB
```

- [ ] Freeze the predicted workspace knee before running the budget sweep.
- [ ] Run each budget in a separate process with the locked kernel profile.
- [ ] Report predicted and observed workspace knees and their relative error.
- [ ] Keep preparation time outside execution and final host D2H inside the
      full application timing.

## Experiment 0B: NUMA-Interleaved Knee Shift

This controlled follow-up tests whether changing only effective host-supply
bandwidth moves the resident-Q knee by the predicted amount.

Locked differences from Experiment 0A:

```text
CPU affinity:          NUMA node5 CPUs, 160-191,416-447
pinned-memory policy:  interleave=5,7
container mem nodes:   5,7
```

Everything else remains fixed, including physical GPU3, the 524,288-token BF16
MHA shape, `kv_chunk=4096`, kernel `128x64/8/3`, buffer counts, host output,
4GiB workspace guardrail, and resident FA4 roof.

The existing bare interleaved measurement is 56.7595545GB/s. Before launching
the new q-sweep:

- [x] Re-measure bare and concurrent H2D on physical GPU3 with pinned pages
      interleaved across nodes 5 and 7.
- [x] Use the same two-copy 112MiB BF16 K/V payload.
- [x] Freeze the concurrent aggregation rule and create a separate prediction
      artifact before inspecting any interleaved q result.
- [x] Compute `q_star=P_FA4/B_P` and `q_95=0.95*q_star` from the newly measured
      concurrent bandwidth.

The preliminary bare-bandwidth estimate gives:

```text
B_P preliminary:    56.7595545 GB/s
q_star preliminary: 3758.3629
q_95 preliminary:   3570.4448
```

The formal GPU3 interleaved calibration used 10 warmups and 50 samples:

| Measurement | Median GB/s | p10 GB/s | p90 GB/s |
|---|---:|---:|---:|
| Bare | 56.7552 | 56.7497 | 56.7587 |
| Concurrent, `q_compute=8192` | 56.7196 | 56.5220 | 56.7473 |
| Concurrent, `q_compute=16384` | 56.7144 | 56.5102 | 56.7347 |

Using the same equal-weight aggregation rule as Experiment 0A:

```text
B_P formal:         56.7170122 GB/s
q_star predicted:   3761.1820
q_95 predicted:     3573.1229
q_star aligned:     3840
q_95 aligned:       3584
workspace at q=3840: 490356736 bytes = 467.640625 MiB
```

The quick visualization sweep is predeclared as:

```text
q = 3072 3328 3456 3584 3712 3840 4096
per q: 1 warmup + 3 measured repeats in one independent process
```

This first pass is exploratory and optimized for turnaround. Publication
replications remain three independent processes with five measured repeats per
process and will be run only after the knee-shift plot is inspected.

The first comparison plot must contain, on the same axes:

- the original single-node measured points;
- the new `interleave=5,7` measured points;
- the original 37.284GB/s exact staircase roof;
- the newly measured interleaved-bandwidth exact staircase roof;
- the shared 213.323TFLOP/s resident FA4 roof;
- both predicted knee markers.

Acceptance signal:

```text
single-node observed knee:   approximately 5.3K-5.5K effective q
interleaved predicted knee:  approximately 3.6K-3.8K effective q
```

The experiment supports the model only if the new transition moves left in the
direction and approximate magnitude predicted from the independently measured
bandwidth increase. Original single-node anomaly replications, including
`q=6272` and `q=7296`, are explicitly scheduled after this quick comparison.

Current quick status:

| q | Pipeline TFLOPS | Status |
|---:|---:|---|
| 3,584 | 196.978 | clean |
| 3,712 | 200.320 | clean |
| 4,096 | 203.347 | clean |
| 3,840 | 135.061 | excluded: progressive host-memory contention |

- [x] Generate the first two-roof comparison plot from the three clean points.
- [ ] Re-run `q=3840` after node7 has at least 16GiB free before allocation.
- [ ] Complete `q=3072`, `3328`, and `3456` in an uncontended host-memory
      window.
- [ ] Avoid interpreting GPU `Exclusive_Process` as host-memory isolation.
- [ ] Consider a reuse-one-allocation multi-q exploratory runner if repeated
      28GiB pinned allocations remain the dominant turnaround cost.

Pause snapshot, 2026-08-24:

- No SeqAttn benchmark or calibration process remains running in
  `seqattn-roofline-gpu3-interleave57`.
- The second `q=3840` rerun was stopped after more than five minutes because it
  was still preparing pinned host tensors: RSS had reached approximately
  24GiB, the main thread was using one CPU core, GPU utilization was 0%, and no
  result JSON had been written.
- A benchmark-only WIP change now issues the Q/K/V pinned allocations in
  parallel, keeps chunk filling on 32 CPU workers, and overlaps output-buffer
  allocation with input preparation.
- Python compilation passed inside the container. The first allocation smoke
  test could not run because physical GPU3 had already been acquired by an
  unrelated process (`PID 3156762`) under `Exclusive_Process`; this is an
  environment block, not a successful validation of the WIP path.
- On resume, first wait for GPU3 to become context-free, then run a small
  pinned-allocation smoke test and a small streaming smoke test before another
  524K-token `q=3840` process.
- The three clean interleaved observations and the two-bandwidth comparison
  figure are checkpointed now; no conclusion depends on the excluded or
  incomplete `q=3840` runs.

## Deferred Robustness Experiments

These are not required for Experiment 0 and must not delay the primary 4K
validation:

- [ ] Repeat the Q sweep with 8K and 16K K/V chunks.
- [ ] Test remote-NUMA placement.
- [ ] Repeat on A30.
- [ ] Extend to GQA, FP16, causal attention, and INT8 K/V.
- [ ] Add a same-kernel GPU-backed SeqAttn roof.
- [ ] Replace planner heuristics with measured hardware profiles.
- [ ] Replicate the original single-node `q=5248`, `q=6272`, and `q=7296`
      points after completing the quick NUMA-interleaved comparison.

## Completion Criteria

Experiment 0 is complete only when:

1. `B_P` and `P_FA4` are independently measured and frozen first.
2. `prediction.json` predates every Q-sweep result.
3. The fixed-4K Q sweep reports an objectively defined observed knee.
4. Timing separates the compute-pipeline metric from full host-output wall
   time.
5. The fixed-4K workspace sweep validates the Q-to-HBM mapping.
6. All raw JSON, commands, environment records, and profiler artifacts are
   retained.
