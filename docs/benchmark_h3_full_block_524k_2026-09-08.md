# Live RTX 5090: 524K full H3 block timing-model validation

Date: 2026-09-08. This is a new GPU experiment, not a replay of the 81K results.

## Result

The frozen full-block prediction was **65.116806 seconds**. After two complete
warmups, three full-block wall times were **60.961586, 60.909482 and 61.018617
seconds**. The median was **60.961586 seconds**, so the model overpredicted by
**4.155220 seconds / 6.816128%**.

```text
signed error = (frozen prediction / measured median - 1) * 100
```

| Phase | Wall seconds | Included in final error? |
|---|---:|---|
| Complete warmup 1 | 63.213186 | No |
| Complete warmup 2 | 60.864518 | No |
| Measured repeat 1 | 60.961586 | Yes |
| Measured repeat 2 | 60.909482 | Yes |
| Measured repeat 3 | 61.018617 | Yes |
| Measured median | 60.961586 | Reference |
| Frozen prediction | 65.116806 | Prediction |

Mean measured wall time was 60.963228 seconds, sample standard deviation
0.054586 seconds and coefficient of variation 0.08954%. The three individual
prediction errors were +6.8161%, +6.9075% and +6.7163%.

This supersedes any expectation that the earlier 81K replay's roughly 1.4%
time error would automatically hold for a live 524K run.

## Workload and environment

| Parameter | Value |
|---|---|
| Physical GPU | GPU 3, NVIDIA GeForce RTX 5090 |
| GPU UUID | `GPU-35d69b8c-779a-bb45-263c-27587354c413` |
| Tokens | 524,288, one non-causal attention segment |
| Block | Real checkpoint block 25 |
| Checkpoint | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Hidden / FFN width | 5376 / 14336 |
| Attention | 56 heads, head dimension 128, BF16 |
| Execution | Materialized, complete production modulated callbacks, no LoRA |
| Q / KV tile | 3840 / 4096 |
| Projection / FFN tile | 4096 / 4096 |
| K/V / projection / output slots | 2 / 2 / 2 |
| RoPE | 96 dimensions |
| Timestep coordinates | Real checkpoint's midpoint adaln basis, width 8 |
| Conditioning | Three modulation rows; all synthetic tokens use row 0 |
| CPU affinity | `160-191,416-447` |
| NUMA memory policy | `interleave=5,7` |
| PyTorch / CUDA | `2.10.0+cu128` / `12.8` |
| ComfyUI / comfy-aimdo | `0.30.0` / `0.4.11` |
| comfy-kitchen | `0.2.26` |

The Docker image was
`sha256:2ae518ea71d53d7c167a1d46b195281d4c6e5bc42b370568bf91280d3ef81385`
(`minimax-h3-seqattn:checks-release-0.4.4`). Both calibration and measurement
containers used that same image and exited successfully.

The actual consumer's `_materialized_block_parts()` supplied the production
callbacks. The timed region includes modulation generation, norm1/modulation,
QKV projection, Q/K norm+RoPE, streamed attention/finalize, output projection,
residual H2D/gating, norm2/modulation, FC1, SwiGLU/FC2, FFN gating and output
D2H. It does not use the older benchmark callbacks that omitted norm1/RoPE.

The checkpoint weights are real; hidden inputs and token positions are fixed
synthetic data. This is a full DiT block benchmark, not end-to-end video generation.
Checkpoint loading, weight preparation, runner/workspace construction, input
population and per-repeat input restoration are outside the timed region.

## Independent prediction and warmup protocol

1. Confirm GPU 3 is idle and fix CPU/NUMA policy.
2. In a dedicated calibration process, load and prepare the actual block weights.
3. Enumerate all operator shapes required by the 524K graph, including tails and
   different transfer payloads at equal token counts.
4. Calibrate each used operator with two warmups and seven CUDA-event repeats.
   Use median latency and maximum observed private allocator workspace.
5. Calibrate pinned H2D while two resident attention updates run concurrently.
   Position transfers use pageable CPU storage, matching that callback path.
6. Save the exact-shape profile and frozen prediction **before** launching the
   full-block evaluation process. No formal block timing is fitted back into it.
7. In a fresh process with the same image, GPU, weights, affinity and memory
   policy, perform two complete block warmups and three complete measured runs.
8. Before every full-block run, restore the same pinned hidden input outside
   timing. Synchronize CUDA at the timing boundary and after the block.
9. Check the entire final output for finiteness outside timing, and compare
   output samples and runtime/model counts across repeats.

Weights were prepared before measurement. NVML-pressure polling and Nsight
profiling were not enabled for primary timing. A separate reset template consumes
5,637,144,576 host bytes; it is benchmark support storage, not part of the modeled
operator's host activation budget.

GPU 3 was idle before launch and returned to 2 MiB / 0% utilization after the
experiment. Another task was active on GPU 0 after calibration; this is retained
as an environment limitation because shared host resources can still matter.

## Correctness and consistency

- Every element of the final hidden output was finite.
- The sampled output signatures were identical across the three measured runs.
- Actual and modeled counts matched: **128 projection tiles, 137 Q chunks,
  17,536 K/V updates, 128 FFN calls**.
- Actual and modeled core workspace budget bytes matched.
- Calibration and evaluation agreed on GPU UUID, framework/kernel versions,
  affinity, memory-node mask, driver source hash and block-weight content hash.
- Core and adapter source files remained unchanged during the experiment.

Mean measured materialization time was 1.232879 seconds; attention plus consumer
execution took 59.729688 seconds. Aggregate timings do not isolate the remaining
6.82% model error. In particular, microbenchmark transfer rates, clock state,
CPU submission gaps and first-K/V initialization can differ from long-run
behavior. The attention calibration measures steady update kernels; the first
K/V tile uses an initialization specialization. No post-hoc correction is added.

Torch's measured allocated peak was 2108.469 MiB and reserved peak was 2448 MiB.
These are process allocator metrics, not an isolated activation-only comparison.

## Evidence and reproduction

The [summary](experiments/h3_full_block_524k_20260908/summary.json),
[raw warmup/measurement records](experiments/h3_full_block_524k_20260908/measurement.json),
[frozen prediction summary](experiments/h3_full_block_524k_20260908/prediction_summary.json),
[exact-shape profile](experiments/h3_full_block_524k_20260908/profile.json), and
[raw operator measurements](experiments/h3_full_block_524k_20260908/operator_measurements.json)
are retained. Source and checkpoint fingerprints are recorded in those artifacts.
The complete frozen prediction and container logs are under
`benchmarks/results/h3_full_524k_20260908/`.

The experiment driver is archived as
[consumer-side source text](experiments/h3_full_block_524k_20260908/consumer_driver.source.txt).
It intentionally runs outside the core package: checkpoint loading and ComfyUI
callbacks remain consumer responsibilities. The working driver, compiled-kernel
cache and full artifacts remain at `/tmp/seqattn-h3-524k-20260908/`.

The two container phases used:

```text
numactl --interleave=5,7 python -u /artifacts/driver.py calibrate
numactl --interleave=5,7 python -u /artifacts/driver.py run
```

They exposed only GPU 3, used the CPU/memory sets above, mounted the checkpoint
and both source repositories read-only, and wrote results only to the experiment
artifact directory. The protocol JSON records the two retained container names
and immutable image ID.
