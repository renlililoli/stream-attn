# H3 DiT runtime contracts

Status: current for `seqattn-core 0.3.0a4` on 2026-08-28.

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
O       = exact_dense_attention(Q, K, V)
X1      = attention_epilogue(O, X0)
X2      = mlp(X1)
```

`attention_epilogue` may include output projection, gating, and residual work.
`mlp` receives and returns complete device tiles. SeqAttn does not prescribe a
checkpoint-specific operator decomposition inside either callback.

## Explicit storage policies

The caller selects one of two runners. There is no automatic policy switch.

| Policy | Runner | Host state | Projection work |
|---|---|---|---|
| Materialized | `H3MaterializedRunner` | One pinned hidden tensor plus complete pinned Q/K/V | Project complete Q/K/V once per block |
| Recompute | `H3RecomputeRunner` | Two distinct pinned hidden tensors; no host Q/K/V | Regenerate Q and K/V for attention tiles |

Materialized execution minimizes repeated projection but uses sequence-sized
host Q/K/V. Recompute reduces host activation memory but repeats K/V projection
for every resident Q pass.

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
  -> complete pinned host Q/K/V
  -> exact attention with device output consumer
  -> attention epilogue
  -> MLP tiles
  -> completed hidden D2H into the original allocation
```

Complete Q/K/V production finishes before attention starts. During attention,
the original hidden values remain available as residual input. A complete raw
attention output and complete post-attention hidden tensor are never stored on
the host.

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
  -> exact attention device output
  -> attention epilogue and MLP
  -> completed hidden D2H into destination
```

The source must remain immutable for the complete block because later query
ranges regenerate K/V from it. `run_blocks_` ping-pongs the two hidden tensors
between blocks and returns whichever tensor contains the final result.

The recompute workspace owns one hidden staging allocation sized to
`max(q_chunk_tokens, kv_chunk_tokens)`. It allocates no complete host Q/K/V.

## Sequence metadata

`H3SequenceMeta.cu_seqlens` must be a one-dimensional CPU `int32` tensor that:

- starts at zero;
- ends at the hidden token count;
- is non-decreasing;
- contains at least two boundaries.

Empty packed segments are allowed. Projection and attention tasks must still
respect every boundary.

## Independent chunk axes

| Axis | Used by | Selection basis |
|---|---|---|
| `projection_tile_tokens` | Materialized QKV producer | Projection kernel saturation and producer workspace |
| `q_chunk_tokens` | Attention resident Q | Measured host-memory roofline and CUDA workspace |
| `kv_chunk_tokens` | Attention K/V tile | Copy/update overlap after Q is calibrated |
| `ffn_tile_tokens` | Attention consumer and FFN | Consumer kernel saturation and auxiliary workspace |

No equality relationship is required. The default H3 consumer configuration
is:

```toml
[minimax_h3]
execution_mode = "materialized"
projection_tile_tokens = 4096
ffn_tile_tokens = 4096
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

This compares logical sequence-sized activation storage only. It excludes
allocator overhead, model weights, callback-owned buffers, CUDA context, and
SeqAttn auxiliary workspaces.

## Reuse and concurrency

H3 runners, their projection runners, and their attention runners are
single-flight. Reuse the stack for compatible serial blocks or denoise steps.
Create separate stacks for true concurrency. Do not overlap calls that share
persistent streams, events, or buffers.

The current materialized-versus-recompute calibration is recorded in
[`benchmark_h3_qkv_recompute_profile_2026-08-27.md`](benchmark_h3_qkv_recompute_profile_2026-08-27.md).
