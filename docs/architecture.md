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

## Planned follow-ups

- Shape-specific autotuning cache for block sizes, warps, stages, and K/V ring
  depth.
- Optional CUDA/CUTLASS backend after Triton profiling identifies kernel-bound
  cases that cannot be resolved by scheduling or fusion.
- Producer/consumer hooks that connect chunked QKV projection and output
  projection without materializing all Q/K/V or attention output in CPU RAM.
- Backward kernels only after the inference API and memory contract stabilize.
