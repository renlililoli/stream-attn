# H3 projection and MLP chunk calibration

Status: current for `seqattn-core 0.3.0a4` on 2026-08-28.

This guide covers the secondary H3 tile settings
`projection_chunk_tokens` and `mlp_chunk_tokens`. Calibrate attention Q and K/V
chunks first with [`q_chunk_calibration.md`](q_chunk_calibration.md).

## Why the axes are separate

Projection and MLP tiles control consumer GEMM efficiency, copy latency, and
temporary activation memory. They do not control how many times attention
scans the complete K/V sequence. Conversely, a measured attention roofline
does not determine the smallest efficient projection or MLP tile.

Recompute execution has no materialized projection subtile: its Q and K/V
callback ranges are fixed by `q_chunk_tokens` and `kv_chunk_tokens`.
`qkv_tile_tokens` therefore affects materialized execution only.

## MLP work model

For hidden width `H`, SwiGLU intermediate width `F`, and chunk size `C`, the two
MLP GEMMs perform:

```text
FC1 FLOPs = 4 * C * H * F
FC2 FLOPs = 2 * C * H * F
MLP FLOPs = 6 * C * H * F
```

The per-chunk time is bounded by the slowest active pipeline stage:

```text
T_chunk(C) ~= max(T_h2d(C), T_gpu(C), T_d2h(C))
```

With measured copy bandwidths `B_in` and `B_out`, element size `s`, fixed copy
latencies `alpha_in` and `alpha_out`, and measured GPU throughput `P(C)`:

```text
T_h2d(C) = alpha_in  + C * H * s / B_in
T_d2h(C) = alpha_out + C * H * s / B_out
T_gpu(C) = alpha_gpu + 6 * C * H * F / P(C) + T_other(C)
```

Bandwidth and plateau throughput show whether copies can be hidden at large C,
but they do not produce a unique minimum tile. The minimum depends on fixed
latency and the measured `P(C)` saturation curve.

## Workspace bound

Increasing `mlp_chunk_tokens` grows H3 auxiliary CUDA storage approximately
linearly with hidden width and element size. The current estimator includes
one MLP device tile and one or two final-output buffers:

```text
consumer bytes ~= (C_mlp + output_buffers * C_output) * H * s
```

Materialized execution additionally owns projection staging:

```text
projection bytes ~= projection_buffers * C_proj * H * s
```

Recompute instead owns hidden staging for the larger attention tile:

```text
recompute staging bytes ~= max(Q_attn, K_attn) * H * s
```

Use runner plan estimates as the SeqAttn-owned CUDA requirement. Add model
weights, CUDA context, allocator reserve, and callback temporaries separately
when enforcing a whole-process target.

## Calibration procedure

1. Freeze model block, dtype, quantization mode, GPU, CPU affinity, NUMA policy,
   attention Q/KV plan, and weight residency behavior.
2. Run every candidate in an independent process or reset the complete input
   state between repeats.
3. Sweep aligned projection and MLP candidates, normally beginning with 2048
   and 4096 tokens.
4. Record full block wall time, projection time, MLP time, CUDA workspace,
   process GPU peak, host activation bytes, and correctness signatures.
5. Retest the winner on an idle GPU in a fresh process.

For materialized execution, use a small grid over `(C_proj, C_mlp)` if both
operators are unsaturated. For recompute, hold the unused materialized
projection setting fixed and sweep only `C_mlp`.

## Selection rule

Choose the smallest aligned tile that satisfies all of the following:

- block wall time is on the measured plateau or within the deployment's
  accepted margin;
- larger tiles do not produce a repeatable independent-process win;
- whole-process GPU memory remains below the target with reserve;
- callback outputs match the reference contract;
- no timing point mixes different weight residency, NUMA, or attention plans.

Do not choose from advertised GPU TFLOPS, nominal PCIe bandwidth, or a single
profile capture. Nsight data explains the result; it is not the primary wall
latency measurement.

## Current H3 default

The 2026-08-27 RTX 5090 block-25 calibration measured 4096-token projection and
MLP tiles as modestly faster than 2048-token tiles while keeping the additional
workspace small for that deployment. The current H3 TOML defaults are
therefore:

```toml
[minimax_h3]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

This is an H3 integration default, not a generic device default. Recalibrate
when GPU, kernel implementation, quantization, block dimensions, weight
residency, or process memory target changes.

Full measurements are in
[`benchmark_h3_qkv_recompute_profile_2026-08-27.md`](benchmark_h3_qkv_recompute_profile_2026-08-27.md).
