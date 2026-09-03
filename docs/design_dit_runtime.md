# H3 DiT runtime contracts

Status: current for `seqattn-core 0.4.0a1` on 2026-09-03.

The H3 runtime schedules a generic transformer block through consumer-provided
callbacks. SeqAttn owns tiling, host/device buffers, attention execution,
stream ordering, and block-to-block hidden-state movement. The consumer owns
model-specific projection, normalization, modulation, residual, MLP, and
weight-lifecycle implementations exposed through those callbacks.

The core package must not import ComfyUI, checkpoint loaders, node classes, or
consumer-specific weight objects.

## Block contract

For one pinned CPU hidden state `X0`, a consumer implements operations
equivalent to:

```text
Q, K, V = projection_and_attention_preprocessing(X0)
O       = configured_attention(Q, K, V)
X1      = attention_epilogue(O, X0)
X2      = mlp(X1)
```

`attention_epilogue` may include output projection, gating, and residual work.
`mlp` receives and returns complete device tiles. SeqAttn does not prescribe a
checkpoint-specific operator decomposition inside either callback.

Dense mode preserves exact attention. `sol_streaming` changes only the attention
line to an explicit block approximation; callback ownership and block order stay
unchanged.

## Explicit storage policies

The caller selects one of two runners. There is no automatic policy switch.

| Policy | Runner | Host state | Projection work |
|---|---|---|---|
| Materialized | `H3MaterializedRunner` | One pinned hidden tensor plus complete pinned Q/K/V | Project complete Q/K/V once per block |
| Recompute | `H3RecomputeRunner` | Two distinct pinned hidden tensors; no host Q/K/V | Regenerate Q and K/V for attention tiles |

Materialized execution minimizes repeated projection but uses sequence-sized
host Q/K/V. Recompute reduces host activation memory but repeats K/V projection
for every resident Q pass.

## Explicit attention policy

`H3Config.attention_mode` is `"dense"` by default and may be set to
`"sol_streaming"`. This is independent from `execution_mode`: materialized and
recompute storage can each execute either dense or Sol attention. It is also
independent from `StreamingAttentionConfig.backend`; Sol V1 requires the built-in
Triton backend.

For every block call in Sol mode, the consumer supplies:

- `H3DenoisingStep(step_index, total_steps)` with zero-based indices;
- a non-negative `block_index`;
- `H3SequenceMeta.exact_prefix_tokens`, one value per packed segment.

With the defaults, `ceil(total_steps * 0.2)` early steps and block indices below
2 stay dense. Remaining blocks use `tau=1.0`. Policy-selected dense warmup is
intentional; missing metadata, causal attention, unsupported hardware/dtype/head
layout, and insufficient workspace raise errors without a compatibility fallback.

Sol V1 is single-GPU BF16 non-causal self-attention on SM80 or newer, with head
dimension 128 and equal Q/KV head counts. Every Q chunk still consumes all K/V.
Materialized execution uses projection-time BF16 summaries and INT8 K/V
transport. Recompute uses BF16 transport and adds one K/V summary projection
pass before routed attention.

## Public callback types

Materialized projection uses `H3MaterializedProjection`:

```text
project_qkv(hidden_tile, start, stop) -> (q, k, v)
```

The callback returns complete CUDA tensors for the projection tile on the
current runner stream.

Recompute uses `H3RecomputeProjection`:

```text
project_q(hidden_tile, destination_q, start, stop) -> None
project_kv(hidden_tile, destination_k, destination_v, start, stop) -> None
```

These callbacks direct-write complete attention tiles. Q ranges follow planned
`q_chunk_tokens`; K/V ranges follow planned `kv_chunk_tokens`. The materialized
`projection_tile_tokens` setting does not affect recompute callback ranges.

Both policies consume `H3BlockOps`:

```text
attention_epilogue(attention_tile, residual_hidden_host, start, stop)
    -> post_attention_device_tile

mlp(post_attention_device_tile, start, stop)
    -> completed_hidden_device_tile
```

Callbacks must enqueue all work on the current CUDA stream before returning.
Returned tensors must cover the complete `[start, stop)` range, remain on the
planned device, and remain valid until consumed.

## Weight leases

Projection and consumer weights have separate optional context factories:

- `H3MaterializedProjection.weight_lease`;
- `H3RecomputeProjection.weight_lease`;
- `H3BlockOps.consumer_lease`.

The projection lease covers the materialized producer phase or the complete
recompute attention scan. The consumer lease covers attention epilogue and MLP
work. SeqAttn calls these contexts but does not implement model-specific
loading, eviction, quantization, or prefetch behavior.

## Materialized execution

`H3MaterializedRunner.run_block_` mutates one pinned hidden allocation in
place:

```text
hidden host source
  -> projection tiles
  -> dense: complete pinned host BF16 Q/K/V
  -> Sol: pinned BF16 Q, BF16 summaries, and INT8 K/V
  -> configured dense or Sol attention with device output consumer
  -> attention epilogue
  -> MLP tiles
  -> completed hidden D2H into the original allocation
```

Complete Q/K/V production finishes before attention starts. During attention,
the original hidden values remain available as residual input. A complete raw
attention output and complete post-attention hidden tensor are never stored on
the host.

In Sol mode, each projection tile is fused with summary generation and INT8 K/V
encoding before D2H. The runner reuses the existing BF16 K/V arena byte storage
for the smaller INT8 payload, so switching between policy-selected dense and Sol
blocks does not require a second sequence-sized K/V allocation. Attention loads
the precomputed summaries once per segment and encoded K/V for each Q chunk.

`run_blocks_` repeats this in-place contract over an iterable of
`(H3MaterializedProjection, H3BlockOps)` pairs and returns the original hidden
tensor.

## Recompute execution

`H3RecomputeRunner.run_block` requires distinct source and destination pinned
hidden tensors:

```text
immutable source hidden tile
  -> Q direct-write for one resident Q range
  -> repeated K/V direct-write for the complete segment
  -> configured dense or Sol attention device output
  -> attention epilogue and MLP
  -> completed hidden D2H into destination
```

The source must remain immutable for the complete block because later query
ranges regenerate K/V from it. `run_blocks_` ping-pongs the two hidden tensors
between blocks and returns whichever tensor contains the final result.

The recompute workspace owns one hidden staging allocation sized to
`max(q_chunk_tokens, kv_chunk_tokens)`. It allocates no complete host Q/K/V.

In Sol mode, recompute performs one extra complete K/V projection pass for the
summary, followed by the normal complete K/V regeneration for each Q chunk.

## Sequence metadata

`H3SequenceMeta.cu_seqlens` must be a one-dimensional CPU `int32` tensor that:

- starts at zero;
- ends at the hidden token count;
- is non-decreasing;
- contains at least two boundaries.

Empty packed segments are allowed. Projection and attention tasks must still
respect every boundary.

`exact_prefix_tokens` is optional for dense execution and mandatory for Sol. Each
value is a token count within its segment. The exact region applies to both Q
and K/V and rounds outward to complete 64-token route blocks.

## Independent chunk axes

| Axis | Used by | Selection basis |
|---|---|---|
| `projection_tile_tokens` | Materialized QKV producer | Projection kernel saturation and producer workspace |
| `q_chunk_tokens` | Maximum attention resident Q; Sol balances 64-token blocks across chunks | Measured host-memory roofline and CUDA workspace |
| `kv_chunk_tokens` | Attention K/V tile | Copy/update overlap after Q is calibrated |
| `ffn_tile_tokens` | Attention consumer and FFN | Consumer kernel saturation and auxiliary workspace |
| 64-token Sol route block | Sparse routing only | Fixed V1 algorithm contract |

No equality relationship is required. The default H3 consumer configuration
is:

```toml
[minimax_h3]
execution_mode = "materialized"
attention_mode = "dense"
projection_tile_tokens = 4096
ffn_tile_tokens = 4096
sol_tau = 1.0
sol_first_dense_step_fraction = 0.2
sol_first_dense_layers = 2
```

`load_h3_config()` reads `SEQATTN_CONFIG` or the default SeqAttn TOML
file. These defaults do not replace topology-specific attention calibration.

See [`design_dit_mlp_chunk_model.md`](design_dit_mlp_chunk_model.md) for
secondary tile calibration and
[`q_chunk_calibration.md`](q_chunk_calibration.md) for attention calibration.

## Capacity tradeoff

For token count `N`, hidden width `H`, Q heads `Aq`, KV heads `Akv`, head
dimension `D`, and element size `s`:

```text
materialized host activation ~= N * s * (H + (Aq + 2*Akv) * D)
recompute host activation    ~= 2 * N * H * s
```

For materialized Sol, the existing BF16 K/V arena storage is reused as the INT8
payload backing and small per-block scale/summary arrays are added. This keeps
dense and Sol available on one runner without another sequence-sized host K/V
allocation.

This compares logical sequence-sized activation storage only. It excludes
allocator overhead, model weights, callback-owned buffers, CUDA context, and
SeqAttn auxiliary workspaces.

The Sol CUDA workspace additionally holds K centroids, V sums, FP32 diagonal K
statistics, Q-block thresholds, and route counters. `build_sol_streaming_plan()`
accounts for these allocations together with the borrowed dense workspace.

## Reuse and concurrency

H3 runners, their projection runners, and their attention runners are
single-flight. Reuse the stack for compatible serial blocks or denoise steps.
Create separate stacks for true concurrency. Do not overlap calls that share
persistent streams, events, or buffers.

Dense and Sol calls constructed by `build_h3_runner()` share one CUDA workspace
and one single-flight lock. Sparse symbols remain outside the multi-GPU plugin
API; V1 has no distributed execution contract.

The current materialized-versus-recompute calibration is recorded in
[`benchmark_h3_qkv_recompute_profile_2026-08-27.md`](benchmark_h3_qkv_recompute_profile_2026-08-27.md).
