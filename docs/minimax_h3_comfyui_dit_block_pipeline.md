# MiniMax-H3 ComfyUI DiT block streaming pipeline

Date: 2026-08-25

## Scope and ownership boundary

This document defines the target dataflow for the MiniMax-H3 DiT block stack
used by the ComfyUI integration. The objective is exact dense attention with a
CPU-backed sequence state while keeping the complete DiT latency within 20% of
the matching ComfyUI-native path. A 15% gap is the follow-up target.

The block scheduler is owned by `seqattn_core`. ComfyUI remains responsible
for model loading, input packing, timestep construction, the final output
layer, and the checkpoint-specific INT8 ConvRot linear operators. It does not
own the block-forward schedule, activation chunk loops, CUDA stream ordering,
or weight-prefetch policy.

The ComfyUI adapter supplies only leaf operations and their weight leases:

- AdaLN projection;
- QKV projection;
- attention output projection;
- MLP FC1 and FC2;
- normalization weights, epsilons, dimensions, and quantization metadata.

SeqAttn owns RMS normalization, modulation, Q/K normalization, RoPE, attention,
SwiGLU, gate/residual epilogues, buffer allocation, chunk reblocking, and all
stream/event dependencies. This preserves the existing optimized quantized
GEMMs without inheriting ComfyUI's block execution structure.

V1 is BF16, batch-one, inference-only execution. Training, dropout, and patches
that change the DiT block mathematical contract are outside this design.

## Block contract

For a pinned CPU hidden state `X0` with shape `[N, H]`, one block computes:

```text
H1       = Norm1(X0)
H1_mod   = H1 * (1 + scale_msa) + shift_msa
Q, K, V  = QKNormRoPE(QKV(H1_mod))
O        = ExactDenseAttention(Q, K, V)
X1       = X0 + gate_msa * OutProj(O)

H2       = Norm2(X1)
H2_mod   = H2 * (1 + scale_mlp) + shift_mlp
G, U     = split(FC1(H2_mod))
M        = FC2(SiLU(G) * U)
X2       = X1 + gate_mlp * M
```

Only `X2` returns to CPU. The full raw attention output, `X1`, FC1 output, and
SwiGLU intermediate are never materialized in host memory.

The completed block writes `X2` back into the same pinned allocation that held
`X0`. The overwrite is safe because QKV projection has consumed all of `X0`
before attention begins, and each destination slice is written only after its
attention residual H2D has completed.

## Independent chunk axes

The block plan has four independent sequence chunk sizes:

| Symbol | Stage | Primary constraint |
|---|---|---|
| `C_proj` | norm1, modulation, QKV, Q/K norm, RoPE | quantized QKV GEMM saturation and producer overlap |
| `Q_attn` | resident-query attention super-block | measured host-memory roofline knee |
| `K_attn` | streamed K/V tile | copy/update overlap and attention kernel efficiency |
| `C_mlp` | norm2, FC1, SwiGLU, FC2, epilogue | quantized MLP saturation and activation memory |

No equality relationship is imposed between these values. In particular,
`C_mlp` is not inherited from `C_proj` or `Q_attn`.

`Q_attn` is an explicit runtime input selected from a calibrated roofline
critical point. It is not inferred from an activation-workspace budget. The
current RTX 5090 points are:

| Host-memory policy | Measured effective bandwidth | Aligned critical Q |
|---|---:|---:|
| single NUMA placement | about 37 GB/s | `5760` |
| NUMA nodes 5 and 7 interleaved | about 56.7 GB/s | `3840` |

The measurement and intersection calculations are recorded in
[`rtx5090_host_memory_roofline_experiment0_2026-08-24.md`](rtx5090_host_memory_roofline_experiment0_2026-08-24.md).

The interleaved value is valid only when that placement is explicitly enabled
and the effective bandwidth is reproduced. The selected Q is authoritative.
The runtime reports the estimated device allocation and lets allocation fail
clearly if the explicit chunks do not fit; it does not silently shrink Q or
turn device memory back into a tuning variable. The reported estimated
workspace bytes are diagnostic only; there is no H3 workspace-budget input or
workspace-driven planner mode.

`C_proj` and `C_mlp` are deployment configuration rather than ComfyUI workflow
parameters. They are read from the shared SeqAttn TOML file selected by
`SEQATTN_CONFIG`, or from `~/.config/seqattn/config.toml` by default:

```toml
[minimax_h3]
qkv_tile_tokens = 4096
mlp_tile_tokens = 4096
```

The ComfyUI node exposes only `Q_attn` and `K_attn`. This keeps hardware
roofline selection explicit while allowing QKV and MLP tile calibration to be
deployed without rewriting workflows.

`C_mlp` must be calibrated with the real ComfyUI INT8 ConvRot FC1 and FC2 path.
The minimum production sweep is `256, 512, 1024, 2048, 4096` tokens. The
selection follows the model in
[`minimax_h3_dit_mlp_streaming_chunk_model.md`](minimax_h3_dit_mlp_streaming_chunk_model.md).

## Sequence metadata

The integration constructs sequence metadata before entering the block stack:

- `position_ids_gpu`: positions required by the H3 rotary embedding;
- `modulation_row_ids_gpu`: one compact row index per token;
- `cu_seqlens`: packed attention segment boundaries;
- static shape metadata: `H`, heads, head dimension, and MLP width.

The token-to-modulation mapping is uploaded once and reused by every block.
Each block computes its six AdaLN tables from `t_emb` once. Pointwise kernels
select rows by `modulation_row_ids_gpu`; Python segment loops and per-segment
device copies are not part of the target path.

## Phase A: fused QKV producer

The producer uses a two-slot H2D/compute/D2H ring:

```text
X0_host[C_proj]
    -> hidden H2D slot
    -> Norm1 + AdaLN modulation
    -> INT8 ConvRot QKV projection
    -> split Q/K/V
    -> Q/K normalization + RoPE
    -> Q_host / K_host / V_host
```

The normalized or modulated hidden tile has GPU-tile lifetime only. Q, K, and
V are copied into persistent pinned backing stores with shape
`[N, heads, head_dim]`. The QKV weight group remains leased across all producer
chunks so quantized-weight preparation is not repeated per tile.

The producer events are:

```text
proj_input_ready[slot]
    -> proj_compute_done[slot]
    -> proj_d2h_done[slot]
```

A slot cannot be reused before `proj_d2h_done`. After the final Q/K/V copy,
`kv_store_ready` becomes the global barrier between the producer and consumer.
This barrier is mandatory: an exact query result cannot be finalized until all
keys and values in its packed segment are available.

The QKV lease can be released after the final projection compute event, while
the last Q/K/V D2H is still draining. Consumer-weight preparation may overlap
that drain.

## Phase B: attention and MLP consumer

The consumer holds OutProj, FC1, and FC2 together for the complete resident-Q
sweep. For each query range `[q0, q1)`:

1. Copy `Q_host[q0:q1]` and the original residual `X0_host[q0:q1]` to GPU.
2. Scan the complete K/V segment through a double-buffered `K_attn` ring.
3. Maintain FP32 online-softmax maximum, sum, and accumulator for resident Q.
4. Finalize raw attention output into a buffer separate from resident Q.
5. Run OutProj for the complete finalized Q range, then apply the attention
   gate/residual into a Q-sized post-attention GPU buffer.
6. Reblock the post-attention range into independent `C_mlp` tiles.
7. Run Norm2, AdaLN, FC1, SwiGLU, FC2, and the MLP gate/residual.
8. Copy only the final hidden tile into its original host location.

The separate raw-output buffer is intentional. It frees the resident-Q buffer
immediately after attention finalize, allowing the H2D stream to prefetch the
next Q and residual while the compute stream executes OutProj and MLP. The
August 18, 2026 61,312-token diagnostic also found the separate-output device
consumer faster than reusing the Q buffer.

OutProj runs over the full Q range so its quantized GEMM is not fragmented by
MLP boundaries. A device-side rechunker then presents `C_mlp` ranges:

- complete MLP ranges within one Q range use views of the post-attention
  buffer;
- a range that crosses a Q boundary uses one bounded `C_mlp` carry buffer;
- the final partial range is executed at its valid size without padding tokens.

This permits `C_mlp` to be smaller or larger than `Q_attn` while preserving
contiguous token order and bounded GPU memory.

## Host and device buffer ownership

Persistent host allocations:

```text
hidden_host  [N, H]                 in-place block input/output
q_host       [N, heads, head_dim]
k_host       [N, heads, head_dim]
v_host       [N, heads, head_dim]
```

There is no `post_attn_host` or MLP-intermediate host allocation.

Persistent device allocations owned by the block runner:

```text
projection_input[2]  [C_proj, H]
resident_q            [Q_attn, heads, head_dim]
kv_ring[2]            K/V [K_attn, heads, head_dim]
running_max/sum       FP32 [Q_attn, heads]
accumulator           FP32 [Q_attn, heads, head_dim]
raw_o                 [Q_attn, heads, head_dim]
attention_residual    [Q_attn, H]
post_attn             [Q_attn, H]
mlp_carry             [C_mlp, H]
final_output[2]       [C_mlp, H]
```

Leaf quantized operators may create bounded tile-local temporaries. They must
not retain sequence-sized tensors after the callback returns. A later kernel
optimization may bind their outputs to runner-owned storage, but that is not a
correctness requirement for V1.

## Stream and event schedule

V1 uses one compute stream, one H2D stream, and one D2H stream. Attention and
MLP do not run on separate compute streams because they would compete for SMs,
Tensor Cores, cache, and HBM bandwidth. Copy/compute overlap is retained.

The steady-state consumer timeline is:

```text
compute: attention[q] -> OutProj/Gate[q] -> MLP tiles[q] -> attention[q+1]
H2D:     KV scan[q]   -> Q/residual[q+1] + initial KV tiles[q+1]
D2H:                       final hidden tiles[q]
```

Required events include:

- `q_residual_ready`: resident Q and the original residual are available;
- `kv_ready[slot]` / `kv_free[slot]`: K/V ring ownership;
- `attention_finalized`: raw O no longer depends on Q or K/V;
- `q_free`: the resident-Q buffer may accept the next query range;
- `post_attn_ready`: attention residual has been consumed into `X1`;
- `residual_free`: the residual input buffer may be overwritten;
- `final_output_ready[slot]` / `final_output_free[slot]`: D2H ring ownership;
- `block_complete`: all final hidden D2H copies have completed.

Writing `X2` over `X0` is legal only after the same token range's residual H2D
is ordered before the compute that produces `X2`. The next block must wait for
`block_complete` before reading the in-place host state.

## Weight residency schedule

Each block uses three weight groups:

```text
W_adaln    AdaLN projection
W_qkv      QKV projection plus small Q/K norm state
W_consumer OutProj + FC1 + FC2 plus small normalization state
```

`W_qkv` remains resident for Phase A, then `W_consumer` remains resident for
Phase B. QKV and consumer weights are not repeatedly prepared per chunk.

When the device-memory guard permits it, the scheduler prepares the next
block's AdaLN/QKV group during the current block's consumer phase. If this
temporary overlap does not fit, preparation occurs synchronously at the block
boundary. The weight adapter may use ComfyUI's low-level model-management and
quantized-linear preparation primitives, but ComfyUI's prefetch queue is not
the scheduling authority.

## Proposed interfaces

The block stack is represented by four core contracts:

```text
H3ChunkPlan
    projection_chunk_tokens
    q_chunk_tokens
    kv_chunk_tokens
    mlp_chunk_tokens
    estimated_workspace_bytes

H3SequenceMeta
    position_ids_gpu
    modulation_row_ids_gpu
    cu_seqlens
    model dimensions

H3BlockOps
    weight lease interface
    adaln_linear
    qkv_linear
    out_linear
    fc1_linear
    fc2_linear
    norm weights and epsilons

H3DiTRunner.run_blocks_
    hidden_host
    block adapters
    t_emb
    sequence metadata
    chunk plan
    optional statistics
```

`run_blocks_` mutates and returns the same pinned host tensor. It owns the block
loop so next-block weight preparation can be scheduled independently of
ComfyUI, but it does not own input embedding/packing or the final H3 output
layer.

## Traffic reduction

For `N` tokens, hidden width `H`, and activation element size `s`, joining the
attention consumer directly to the MLP removes:

```text
post-attention D2H = N * H * s
MLP input H2D      = N * H * s
saved per block    = 2 * N * H * s
```

At `N=157249`, `H=5376`, and BF16, one hidden direction is approximately
1.575 GiB. The joined block therefore removes approximately 3.149 GiB of
logical PCIe traffic per block, or 157.46 GiB across 50 blocks in one denoise
step.

The V1 baseline deliberately retains Q/K/V host storage and repeated K/V
scans. Retaining one producer-generated Q range on GPU can remove one Q
D2H/H2D pair, but it complicates producer/Q alignment and output ordering. It
is a V1.1 optimization and is disabled in the baseline design.

## Deferred host-memory-minimal QKV recomputation

Exact dense attention does not fundamentally require materialized host Q/K/V.
A more aggressive capacity mode can retain only CPU-backed block input and
output hidden states, then recompute Q and K/V from the input hidden state as
the attention sweep needs them. This is an inference-time recomputation mode,
not an approximate attention algorithm.

For source and destination pinned hidden tensors, the dataflow would be:

```text
src_hidden_host [N, H]                    read-only for the complete block
dst_hidden_host [N, H]                    final block output

for each resident query range [q0, q1):
    Q = project_Q(src_hidden_host[q0:q1])
    initialize online-softmax state

    for each K/V range [k0, k1):
        hidden_tile = H2D(src_hidden_host[k0:k1])
        K, V = project_KV(hidden_tile)
        update exact attention state directly from device K/V

    finalize attention
    run OutProj, attention epilogue, MLP, and MLP epilogue
    D2H the final range into dst_hidden_host[q0:q1]

swap(src_hidden_host, dst_hidden_host) after the complete block
```

The source hidden tensor cannot be overwritten as query ranges finish. Every
later query range still needs the original block input to regenerate K/V for
all token ranges. Writing an early `X2` range over `X0` would make later K/V
projections read a mixture of block input and block output. Therefore the
practical minimum is two hidden tensors, not one in-place hidden tensor. Using
one physical hidden allocation would require an additional full-output backing
store such as NVMe and would only move, not remove, that storage requirement.

Let `A = heads * head_dim` be the attention width. Ignoring small metadata and
bounded device tiles, the V1 materialized path requires approximately:

```text
M_materialized = N * (H + 3A) * s
```

The recomputation path requires approximately:

```text
M_recompute = N * (2H) * s
```

For MiniMax-H3, `H=5376`, `A=7168`, and BF16. The host-activation comparison
is:

| Tokens | hidden plus Q/K/V | two hidden tensors | Reduction |
|---:|---:|---:|---:|
| `61,312` | 3.07 GiB | 1.23 GiB | 1.84 GiB |
| `157,249` | 7.87 GiB | 3.15 GiB | 4.72 GiB |
| `262,720` | 13.15 GiB | 5.26 GiB | 7.89 GiB |
| `400,000` | 20.03 GiB | 8.01 GiB | 12.02 GiB |

These values exclude model weights, framework state, metadata, and bounded
projection and attention workspaces. The reduction is nevertheless large
enough to make recomputation a plausible capacity mode for host-memory-limited
262K-token and larger workloads.

The cost is repeated projection. If `R = ceil(N / Q_attn)`, each resident-Q
range scans the complete source hidden state and repeats Norm1, modulation,
quantized projection, K normalization, and RoPE. With the current fused INT8
ConvRot QKV operator, collecting Q separately may also compute and discard K/V,
while each K/V scan computes and discards Q. A future Q-only/KV-only packed
operator could reduce this redundant arithmetic, but it is not assumed by this
design.

The recomputation mode would also use a different Q planning objective. The V1
materialized path selects the smallest Q at or above the measured roofline
knee, subject to memory constraints. Recomputation cost grows with the number
of Q ranges, so its planner would instead favor the largest resident Q that
fits the complete block workspace. A mode requiring dozens of full hidden
projection scans is not an acceptable default latency path.

Immediate attention-to-MLP consumption additionally requires QKV, OutProj,
FC1, and FC2 weights to remain available across the query sweep. If those
weight groups cannot coexist under the device-memory guard, the alternatives
are repeated weight preparation or a two-phase path that materializes the
post-attention hidden tensor. Both weaken the latency benefit and must be
measured explicitly.

This recomputation mode is deferred. The V1 implementation order is:

1. Implement the materialized-QKV block runner specified in the preceding
   sections.
2. Join attention finalize, OutProj, both residual epilogues, and the complete
   SwiGLU MLP so only final `X2` returns to host.
3. Validate correctness, stream lifetimes, weight leases, the four independent
   chunk axes, and the `1.20x` latency acceptance target.
4. Evaluate QKV recomputation as a separate capacity experiment only after the
   baseline fused block runner is complete and measured.

The initial `H3DiTRunner` therefore materializes pinned Q/K/V and retains the
single-hidden in-place contract. It must not delay or complicate the baseline
implementation to support recomputation. A later extension may introduce
explicit `materialized`, `kv_only`, and `recompute` storage policies after the
required projection and weight-residency measurements exist.

## Correctness and fallback rules

- Every query range scans all K/V ranges in its packed segment before finalize.
- No Q, K/V, carry, or output slot is reused before its free event.
- MLP reblocking changes scheduling only; the MLP remains token-local and
  mathematically identical to native execution.
- Host in-place writes never race a residual H2D read of the same token range.
- Unsupported training state, custom DiT patches, leaf output shapes, dtypes,
  or quantization formats cause a whole-block or whole-stack native fallback;
  the runtime does not execute a partially compatible graph.
- Performance records include all four explicit chunk sizes so results do not
  accidentally compare different roofline or tile policies as if they were
  equal.

## Experiment and acceptance plan

Correctness tests:

1. Compare one BF16 block with native execution on small deterministic shapes.
2. Cover non-divisible `C_proj`, `Q_attn`, `K_attn`, and `C_mlp` boundaries.
3. Cover an MLP tile that crosses a Q boundary and a final partial carry.
4. Cover multiple modulation rows and packed-sequence boundaries.
5. Stress asynchronous in-place writeback without global synchronization
   between individual chunks.
6. Repeat parity through several consecutive blocks to detect accumulated
   state or lifetime errors.

Performance experiments:

1. Sweep the real INT8 ConvRot QKV path to select `C_proj`.
2. Validate the calibrated `Q_attn` at the active NUMA placement and measure
   neighboring aligned values.
3. Sweep `K_attn` for K/V copy/update overlap.
4. Sweep `C_mlp` at `256, 512, 1024, 2048, 4096` using the full fused MLP path.
5. Profile one block and confirm that post-attention host traffic and host MLP
   intermediate traffic are both zero.
6. Compare at least three warmed complete 50-block DiT runs with the same GPU,
   NUMA placement, INT8 checkpoint, input, and no artificial native allocator
   cap.

The release acceptance target is median streaming DiT time no greater than
`1.20x` native on a sequence for which native completes. The follow-up target
is `1.15x`. Large sequences where native OOMs remain capacity tests and are not
used as the primary latency ratio.

## Initial implementation validation

The first materialized-QKV fused-block validation was run on 2026-08-25 with
the real MiniMax-H3 INT8 ConvRot checkpoint in
`diffsynth:cu128-roofline-fa4-20260824`. The process used physical RTX 5090
GPU 3, CPU node 5, and explicit host-memory interleave across NUMA nodes 5 and
7. Both paths used `Q_attn=3840`, `K_attn=4096`, `C_proj=2048`, and
`C_mlp=2048`; the old path was instrumented to reject the run unless its
effective attention plan also reported `Q_attn=3840`.

For a warmed one-real-block packed input, with two warmups and five measured
forwards, the old split attention/MLP path had a median latency of `170.48 ms`.
The fused block runner had a median latency of `165.38 ms`, about 3.0% faster.
Explicit pinned hidden allocation per forward fell from `167,215,104` bytes to
`83,607,552` bytes. Materialized Q/K/V host storage remained `334,430,208`
bytes, as required by the V1 design. Peak PyTorch device allocation increased
from `1,912,821,248` bytes to `1,956,206,080` bytes because the fused runner
keeps bounded MLP carry and final-output slots on device.

This focused result validates the no-regression target and the intended hidden
memory reduction for the fused block itself. It does not replace the release
requirement for repeated complete 50-block DiT measurements.
