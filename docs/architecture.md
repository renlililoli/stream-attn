# Architecture notes

## Memory hierarchy

The operator has two tiling levels:

- `q_chunk_tokens` is an HBM-resident query super-block.  It controls the FP32
  online-softmax accumulator and therefore most of the workspace footprint.
- `kv_chunk_tokens` is a streamed K/V tile.  Two buffers permit copy/compute
  overlap; a third buffer is available for systems where the copy engine needs
  more queue depth.

Within a K/V tile, the Triton kernel uses `BLOCK_M x BLOCK_N` attention tiles
and never writes the score matrix to global memory.

## Fused recurrence

Each K/V update maintains, per query/head, FP32 state:

```text
m = running row maximum
l = running softmax normalizer
a = running unnormalized output
```

For a new tile with scores `S`, the kernel computes the merged maximum,
rescales the old state, accumulates `exp(S - m) @ V`, and stores the updated
state in one launch.  Final normalization and cast are a separate fused kernel
executed once per query super-block.

## Pipeline invariants

- H2D never overwrites a K/V slot until its compute-free event fires.
- Q is not replaced until every K/V update using it has completed.
- A GPU output slot is not reused until its D2H-free event fires.
- Packed sequence boundaries are scheduler boundaries; no query super-block or
  K/V scan crosses a segment.
- Causal positions use bottom-right alignment for unequal Q/K lengths.

## Projected inference pipeline

`ProjectedAttentionRunner` adds model-projection producer/consumer hooks around
the Triton attention runtime:

```text
CPU hidden
    -> double-buffered hidden H2D
    -> GPU QKV projection callback
    -> Q/K/V D2H into persistent pinned backing buffers
    -> global K/V readiness barrier
    -> Q-resident / KV-streamed Triton attention
    -> GPU output-projection callback
    -> projected output D2H
```

The readiness barrier is required for exact global self-attention: an early
query cannot be finalized until projection has produced every key/value token
in its packed segment.  Projection H2D, projection compute, and Q/K/V D2H are
still pipelined across chunks before that barrier.

The important fusion is on the consumer side.  Raw attention output remains on
GPU and is passed directly to output projection.  Compared with a staged
implementation, this removes exactly one raw-attention D2H and one matching H2D
for every query token.  The callback may also include inference-only epilogues
such as gate/residual application.

QKV and output projection are callbacks rather than hard-coded matmul kernels.
This permits ordinary BF16/FP16 linear layers, quantized modules, model-specific
Q/K normalization, and rotary embedding while the attention core remains
Triton.  The caller is responsible for keeping projection weights resident for
the duration of each phase.

## Planned follow-ups

- Shape-specific autotuning cache for block sizes, warps, stages, and K/V ring
  depth.
- Optional CUDA/CUTLASS backend after Triton profiling identifies kernel-bound
  cases that cannot be resolved by scheduling or fusion.
- Optional prefetched residual/epilogue buffers so model-specific residual H2D
  overlaps the final K/V scan.
- Projection callback variants that can write into persistent GPU output slots
  and avoid temporary allocator traffic for standard dense linear layers.
- Backward kernels only after the inference API and memory contract stabilize.
