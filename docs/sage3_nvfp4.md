# SageAttention3 NVFP4 on RTX 50-series

This is an explicitly approximate, inference-only backend for the extreme-speed
CUDA 13 experiment. Inputs/outputs remain BF16 or FP16; Sage3 internally uses
NVFP4 data and FP8 block scales. Across K/V partitions, SeqAttn keeps normalized
output and LSE in FP32. It does not reduce the accumulated state to BF16.

## Selection and compatibility

- Canonical backend: `sage3`; alias: `nvfp4`.
- `SEQATTN_AUTO_NVFP4=1` enables automatic selection on SM120. The CUDA 13 image
  sets it by default. Ordinary installations retain their previous defaults.
- Supported: equal Q/KV head counts, head dimension 64 or 128, noncausal
  contiguous and projected/recomputed streaming, including H3 device consumers.
- Unsupported auto geometry and causal execution fall back to the existing
  backend. An explicit unsupported Sage3 request fails before execution.
- Explicit backend arguments/environment/TOML preserve their existing precedence.
- Sol, paged/NVMe and plugin-specific execution retain their own contracts;
  this is not a claim that their kernels now use NVFP4.

The backend requires the Sage3 binary patched by the consumer's
`docker/sage3/patch.py` and checks `fp4attn_cuda.seqattn_lse_abi() == 1`.
An unpatched upstream binary cannot be used for partition merging.

## Partition correctness

The pinned upstream Sage3 allocates a returned LSE tensor but comments out its
normal-row writes. The patch writes natural-log LSE from the completed softmax
state, removing its internal `448 * 6` probability scale. Analytic constant-K
cases, including non-aligned tails, verify the resulting LSE.

Q quantization is cached once per Q super-block. K is cloned before smoothing,
and the removed `Q @ mean(K) * scale` score bias is restored in FP32 to give
partitions a common LSE origin. Actual K lengths mask padding. Caller inputs
remain unchanged. The BF16 single-partition regression matches the Sage3 API bitwise;
FP32 merging matches an independent LSE-weighted reference.

Quantization is still lossy and partition-dependent. Equal partition mechanics
do not imply equality with dense BF16 or with differently partitioned NVFP4.

## Workspace

`AttentionPlan.backend_workspace_bytes` reserves a conservative bound for
Sage3's quantization, layout/padding, correction and partial output allocations.
The normal workspace estimate and budget include this reservation. It is not
reported as a new persistent tensor. Query caches are released before the H3
consumer executes, and runners remain single-flight.

The offline H3 estimator currently represents the original online-softmax graph,
not Sage3's query preparation and normalized-LSE execution. Binding an NVFP4 plan
to that estimator fails explicitly; no old profile or accuracy percentage is
reused to claim NVFP4 timing/memory accuracy.

## Measured result, September 9, 2026

Both paths were measured on RTX 5090 GPU 3 in CUDA 13, with identical shapes and
independent processes. The reference uses the tuned Blackwell Triton launch
`(block_m=128, block_n=64, warps=8, stages=3)`.

| Scope | Tuned BF16 Triton | Sage3 NVFP4 |
|---|---:|---:|
| Q3840/K4096/H56/D128 KV update | 2.0524 ms | 1.1453 ms |
| Full 65536-token H3 materialized block | 1.2204 s | 0.8462 s |

The update includes K/V preprocessing, LSE correction and FP32 merge; Q is
prepared once and reused across K/V partitions. Full-block timing includes
query preparation and all real production callbacks. There are five warmups
and twenty update repeats, and two warmups plus three full-block repeats.

Full-block output versus BF16: **NRMSE 18.69%, cosine 0.98288**. This is a
substantial approximation observed in the requested extreme-speed
experiment, not a lossless or final-video-quality claim. All outputs were finite.

The earlier portable-launch diagnostic (32/64/4/2) is retained as evidence but
excluded from the comparison. Only the tuned BF16 reference is used above.
Raw observations and source hashes are in
[`experiments/sage3_nvfp4_20260909`](experiments/sage3_nvfp4_20260909/manifest.json).
