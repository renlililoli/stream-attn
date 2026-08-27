# MiniMax-H3 ComfyUI DiT block streaming pipeline

Date: 2026-08-27

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
- complete QKV projection for materialized execution;
- Q-only and K/V-only direct-write projection for recompute execution;
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

The storage policy is selected explicitly by the caller:

- the materialized runner writes `X2` back into the pinned allocation that
  held `X0`; complete QKV projection has consumed `X0` before attention begins;
- the recompute runner keeps `X0` immutable for the complete block and writes
  `X2` into a distinct pinned destination because later query ranges must still
  regenerate K/V from the original source.

There is no automatic host-memory policy or compatibility wrapper between the
two modes. The implementation details and public projection contracts are also
summarized in
[`dit_qkv_recompute_architecture.md`](dit_qkv_recompute_architecture.md).

## Independent chunk axes

The materialized block plan has four independent sequence chunk sizes:

| Symbol | Stage | Primary constraint |
|---|---|---|
| `C_proj` | norm1, modulation, QKV, Q/K norm, RoPE | quantized QKV GEMM saturation and producer overlap |
| `Q_attn` | resident-query attention super-block | measured host-memory roofline knee |
| `K_attn` | streamed K/V tile | copy/update overlap and attention kernel efficiency |
| `C_mlp` | norm2, FC1, SwiGLU, FC2, epilogue | quantized MLP saturation and activation memory |

No equality relationship is imposed between these values. In particular,
`C_mlp` is not inherited from `C_proj` or `Q_attn`.

The recompute plan has only three sequence axes: `Q_attn`, `K_attn`, and
`C_mlp`. It has no `C_proj`. Every Q-only projection range is exactly one
attention Q range, and every K/V-only projection range is exactly one
attention K/V tile, including non-divisible tails. Its hidden staging capacity
is `max(Q_attn, K_attn)` tokens.

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

For materialized execution, `C_proj` and `C_mlp` are deployment configuration
rather than ComfyUI workflow parameters. They are read from the shared SeqAttn
TOML file selected by `SEQATTN_CONFIG`, or from
`~/.config/seqattn/config.toml` by default:

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

The integration constructs attention boundaries before entering the block stack:

- `cu_seqlens`: packed attention segment boundaries;
- static shape metadata: `H`, heads, head dimension, and MLP width.

Position IDs and the token-to-modulation mapping are adapter-owned device
tensors captured by the projection and consumer callbacks. They are not runner
metadata because SeqAttn does not read them. Each block computes its six AdaLN
tables from `t_emb` once; pointwise callbacks select the required rows without
Python segment loops or per-segment device copies.

## Materialized Phase A: fused QKV producer

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

## Shared attention and MLP consumer

Both storage policies enter the same Triton online-softmax, finalize, and
output-consumer loop. Materialized execution loads Q/K/V from pinned host
backing through the H2D stream and its double-buffer events. Recompute execution
stages the matching source-hidden range and invokes a Q-only or K/V-only
projector directly into the attention workspace on the compute stream.

The consumer holds OutProj, FC1, and FC2 together for the complete resident-Q
sweep. For each query range `[q0, q1)`:

1. Load or recompute Q for `[q0:q1)` and copy the explicit residual
   `X0_host[q0:q1]` to GPU.
2. Scan the complete K/V segment through a double-buffered `K_attn` ring.
3. Maintain FP32 online-softmax maximum, sum, and accumulator for resident Q.
4. Finalize raw attention output into a buffer separate from resident Q.
5. Run OutProj for the complete finalized Q range, then apply the attention
   gate/residual into a Q-sized post-attention GPU buffer.
6. Reblock the post-attention range into independent `C_mlp` tiles.
7. Run Norm2, AdaLN, FC1, SwiGLU, FC2, and the MLP gate/residual.
8. Copy only the final hidden tile into the selected destination host range.

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

Materialized persistent host allocations:

```text
hidden_host  [N, H]                 in-place block input/output
q_host       [N, heads, head_dim]
k_host       [N, heads, head_dim]
v_host       [N, heads, head_dim]
```

There is no `post_attn_host` or MLP-intermediate host allocation.

Recompute persistent host allocations:

```text
source_hidden_host       [N, H]     immutable complete-block input
destination_hidden_host  [N, H]     complete-block output
```

Recompute does not allocate host Q, K, or V. `run_blocks_` swaps the two
physical hidden allocations after each block and returns the one containing
the final result.

Materialized-specific device allocation:

```text
projection_input[2]  [C_proj, H]
```

Recompute-specific device allocation:

```text
hidden_staging  [max(Q_attn, K_attn), H]
```

Shared attention and consumer allocations owned by the runners:

```text
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

Materialized source events include:

- `q_residual_ready`: resident Q and the original residual are available;
- `kv_ready[slot]` / `kv_free[slot]`: K/V ring ownership;
- `attention_finalized`: raw O no longer depends on Q or K/V;
- `q_free`: the resident-Q buffer may accept the next query range;
- `post_attn_ready`: attention residual has been consumed into `X1`;
- `residual_free`: the residual input buffer may be overwritten;
- `final_output_ready[slot]` / `final_output_free[slot]`: D2H ring ownership;
- `block_complete`: all final hidden D2H copies have completed.

The recompute source replaces Q/K/V host-transfer ownership with
`hidden_ready` and `hidden_free` around its single staging allocation. The
compute stream invokes the projector after `hidden_ready`; `hidden_free` is
recorded after the projector has enqueued all work consuming the staged hidden
tile. Q/K/V destinations are then consumed on the same compute stream.

Writing `X2` over `X0` is legal only after the same token range's residual H2D
is ordered before the compute that produces `X2`. The next block must wait for
`block_complete` before reading the in-place host state. This rule applies to
materialized execution; recompute always writes a distinct destination.

## Weight residency schedule

Each block uses projection and consumer weight groups:

```text
W_projection  AdaLN, QKV or Q-only/K/V-only projection, Q/K norm state
W_consumer    OutProj + FC1 + FC2 plus small normalization state
```

For materialized execution, `W_projection` remains leased for the complete
host-QKV producer phase, then `W_consumer` remains leased for the attention and
MLP consumer phase. For recompute, both leases cover the complete attention
sweep because every Q and K/V tile is projected on demand while finalized
query ranges immediately enter the consumer. Weights are not repeatedly
prepared per tile.

When the device-memory guard permits it, the scheduler prepares the next
block's AdaLN/QKV group during the current block's consumer phase. If this
temporary overlap does not fit, preparation occurs synchronously at the block
boundary. The weight adapter may use ComfyUI's low-level model-management and
quantized-linear preparation primitives, but ComfyUI's prefetch queue is not
the scheduling authority.

## Implemented interfaces

The block stack is represented by explicit materialized and recompute
contracts:

```text
H3MaterializedPlan
    projection_chunk_tokens
    q_chunk_tokens
    kv_chunk_tokens
    mlp_chunk_tokens
    estimated_workspace_bytes

H3RecomputePlan
    q_chunk_tokens
    kv_chunk_tokens
    mlp_chunk_tokens
    hidden_staging_tokens
    estimated_workspace_bytes

H3SequenceMeta
    cu_seqlens

H3MaterializedProjection
    project_qkv
    weight_lease

H3RecomputeProjection
    project_q
    project_kv
    weight_lease

H3BlockOps
    attention_epilogue
    mlp
    consumer_lease

H3MaterializedRunner.run_blocks_(hidden_host, sequence_meta, blocks, ...)
H3RecomputeRunner.run_blocks_(hidden_host, scratch_hidden_host,
                              sequence_meta, blocks, ...)
```

`H3MaterializedRunner.run_blocks_` mutates and returns the same pinned host
tensor. `H3RecomputeRunner.run_blocks_` requires two distinct pinned hidden
tensors, ping-pongs them across blocks, and returns the physical tensor holding
the final result. Both own the block loop but do not own input
embedding/packing or the final H3 output layer.

The attention epilogue always receives the residual host tensor explicitly.
The materialized runner passes its current in-place hidden tensor; the
recompute runner passes its immutable source tensor. `H3BlockOps` does not
branch on storage policy.

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

The materialized policy deliberately retains Q/K/V host storage and repeated
K/V scans. Retaining one producer-generated Q range on GPU could remove one Q
D2H/H2D pair, but it would complicate producer/Q alignment and output ordering
and is not implemented. Recompute instead bypasses host Q/K/V completely.

## Implemented host-memory-minimal QKV recomputation

Exact dense attention does not require materialized host Q/K/V. The implemented
recompute runner retains CPU-backed block input and output hidden states, then
regenerates Q and K/V from the immutable input hidden state as the attention
sweep needs them. This is an inference-time recomputation policy, not an
approximate attention algorithm.

For source and destination pinned hidden tensors, the dataflow is:

```text
src_hidden_host [N, H]                    read-only for the complete block
dst_hidden_host [N, H]                    final block output

for each resident query range [q0, q1):
    hidden_q = H2D(src_hidden_host[q0:q1])
    project_Q(hidden_q, attention_q_workspace, q0, q1)
    initialize online-softmax state

    for each K/V range [k0, k1):
        hidden_tile = H2D(src_hidden_host[k0:k1])
        project_KV(hidden_tile,
                   attention_k_workspace,
                   attention_v_workspace,
                   k0, k1)
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

The cost is repeated K/V projection. If `R = ceil(N / Q_attn)`, each resident-Q
range scans the complete source hidden state and repeats Norm1, modulation,
K/V-only quantized projection, K normalization, and RoPE. Q projection runs
once per planned Q range. K/V projection runs once per attention K/V tile for
each query sweep. No projection sub-tiling is allowed inside the core runner.

The public direct-write projection contracts are:

```text
QTileProjector(hidden_tile, destination_q, start, stop) -> None
KVTileProjector(hidden_tile, destination_k, destination_v,
                start, stop) -> None
```

Callbacks are responsible for the complete large-tile fused operation and
writing the supplied destination. The core intentionally does not depend on
ComfyUI or comfy-kitchen. A MiniMax-H3 adapter can use a ConvRot-specialized
operator that selects only Q rows or the contiguous K/V rows, together with
the matching per-output scales, optional bias, and ConvRot metadata.

Recomputation has a different Q planning tradeoff from materialized execution.
Its repeated K/V work grows with the number of Q ranges, so a deployment may
favor a larger resident Q when device memory permits. The caller supplies the
mode and plan explicitly; the core does not infer either from available host
or device memory.

The standalone ConvRot benchmark validates its Q-only and K/V-only row slices
against the complete block-25 QKV projection element by element before timing.
Its formal A/B comparison uses separate processes with the same
`tokens=262720`, `Q_attn=16384`, `K_attn=4096`, and `C_mlp=4096`; only the
materialized path has `C_proj=4096`. It records wall-time median/mean, PyTorch
allocated/reserved peaks, PID NVML peak, RSS peak, and logical host activation.
It runs in the Community ComfyUI `comfyui:cu128` environment pinned to ComfyUI
`0.30.0` commit `9a9fdb10ed144ce760d9682cb247526ea23cc525`, Torch
`2.10.0+cu128`, CUDA `12.8`, and comfy-aimdo `0.4.11`. The benchmark validates
those versions, uses `/opt/ComfyUI/models`, and loads the checkpoint through
`comfy.sd.load_diffusion_model`; the DiffSynth service is not the authoritative
environment for this comparison.

Results from the removed projection-subtiled recompute implementation are
invalid for this architecture. In particular, earlier runs with `Q=196608`,
or any run where recompute callback ranges were controlled by `C_proj`, must
not be used in final performance conclusions.

## Correctness and fallback rules

- Every query range scans all K/V ranges in its packed segment before finalize.
- No Q, K/V, carry, or output slot is reused before its free event.
- MLP reblocking changes scheduling only; the MLP remains token-local and
  mathematically identical to native execution.
- Materialized host in-place writes never race a residual H2D read of the same
  token range.
- Recompute source hidden remains unchanged for the complete block, and source
  and destination may not alias.
- Q-only callback count and ranges equal the planned attention Q chunks;
  K/V-only callback count and ranges equal the attention K/V tiles actually
  scanned, including packed segments, empty segments, and tail chunks.
- Unsupported training state, custom DiT patches, leaf output shapes, dtypes,
  or quantization formats cause a whole-block or whole-stack native fallback;
  the runtime does not execute a partially compatible graph.
- Materialized performance records include all four explicit chunk sizes;
  recompute records include Q, K/V, and MLP chunks and has no projection chunk.

## Experiment and acceptance plan

Correctness tests:

1. Compare materialized and recompute BF16 blocks with full GPU execution on
   deterministic shapes.
2. Cover packed sequences, empty segments, non-divisible Q/K/V tails, and
   different Q and K/V chunk sizes.
3. Assert exact Q-only and K/V-only projector ranges and prove they are
   independent of materialized `C_proj`.
4. Cover an MLP tile that crosses a Q boundary and a final partial carry.
5. Verify materialized asynchronous in-place writeback and recompute source
   immutability plus distinct destination storage.
6. Repeat recompute through multiple blocks and verify the returned physical
   ping-pong buffer.
7. Verify recompute allocates no host Q/K/V and has bounded allocator growth
   across repeated runs.

Performance experiments:

1. Sweep the real INT8 ConvRot materialized QKV path to select `C_proj`.
2. Validate the calibrated materialized `Q_attn` at the active NUMA placement
   and measure neighboring aligned values.
3. Sweep `K_attn` for K/V copy/update overlap.
4. Sweep `C_mlp` at `256, 512, 1024, 2048, 4096` using the full fused MLP path.
5. Profile one block and confirm that post-attention host traffic and host MLP
   intermediate traffic are both zero.
6. Run the formal independent-process materialized/recompute ConvRot A/B with
   the fixed 262,720-token configuration and mandatory small-tile parity.
7. Compare at least three warmed complete 50-block DiT runs with the same GPU,
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
