# Projection-attention-output pipeline benchmark

## Question

The original staged out-of-core path materializes raw attention output in CPU
RAM before output projection:

```text
Triton attention -> raw attention D2H -> CPU -> raw attention H2D -> out_proj
```

The projected pipeline keeps the raw output on GPU:

```text
Triton attention -> out_proj -> projected output D2H
```

This benchmark measures the latency, process GPU-memory peak, and logical PCIe
traffic removed by that consumer-side fusion.  It does not claim that exact
self-attention can begin before all K/V are available.  QKV projection remains
a chunked producer phase followed by a global K/V readiness barrier.

## Compared modes

- `pipeline`: `ProjectedAttentionRunner`; GPU attention output is passed
  directly to the output-projection callback.
- `staged`: the same chunked QKV projection and Triton attention plan, but raw
  attention is copied to pinned CPU RAM and later copied back to GPU for output
  projection.

Both modes use the same random seed, dense BF16 projection weights, H3 tensor
dimensions, packed-sequence layout, attention tiles, and output projection.

## Environment and protocol

- Date: 2026-08-18.
- GPU: NVIDIA GeForce RTX 5090.
- PyTorch: 2.10.0+cu128.
- CUDA runtime: 12.8.
- Shape: hidden size 5,376; 56 heads; head dimension 128.
- Attention: non-causal, one packed segment.
- Projection chunk: 2,048 tokens for 15,104 and 61,312 tokens.
- K/V chunk: 4,096 tokens for 15,104 and 61,312 tokens.
- Attention workspace: 2 GiB for 15,104 and 61,312 tokens.
- Each mode runs in an independent process.
- One warmup precedes the measured repetitions.
- NVML sampling is performed in a separate untimed run at a 10 ms interval.
- 61,312-token points use an 8,192 MiB whole-process target and two measured
  repetitions.  Smaller smoke points use three repetitions without a target
  allocator limit.

These are system-characterization measurements without error bars or
statistical-significance claims.

## Results

| Tokens | Mode | Mean latency | NVML process peak | Logical H2D | Logical D2H |
|---:|---|---:|---:|---:|---:|
| 3,072 | pipeline | 12.35 ms | 1,386 MiB | 157.5 MiB | 157.5 MiB |
| 3,072 | staged | 15.16 ms | 1,596 MiB | 199.5 MiB | 199.5 MiB |
| 15,104 | pipeline | 88.95 ms | 2,718 MiB | 774.4 MiB | 774.4 MiB |
| 15,104 | staged | 102.15 ms | 3,758 MiB | 980.9 MiB | 980.9 MiB |
| 61,312 | pipeline | 843.44 ms | 3,848 MiB | 6.344 GiB | 3.069 GiB |
| 61,312 | staged | 919.79 ms | 7,108 MiB | 7.163 GiB | 3.888 GiB |

Derived comparisons:

| Tokens | Pipeline latency reduction | NVML peak reduction | Raw-attention round trip removed |
|---:|---:|---:|---:|
| 3,072 | 18.6% | 210 MiB | 84.0 MiB |
| 15,104 | 12.9% | 1,040 MiB | 413.0 MiB |
| 61,312 | 8.3% | 3,260 MiB | 1.637 GiB |

The removed round trip is exactly:

```text
2 * tokens * heads * head_dim * element_size
```

For 61,312 BF16 tokens this is:

```text
2 * 61,312 * 56 * 128 * 2 bytes = 1,757,937,664 bytes
```

The observed GPU-peak reduction is larger than the raw tensor size because the
staged path can retain multiple asynchronous raw-attention and projected-output
allocations until their stream work completes.  The projected runner bounds
these lifetimes with reusable output slots and D2H completion events.

## Interpretation

The consumer-side pipeline improves the property that matters for an 8GB video
inference target: intermediate attention output no longer competes with model
weights, the FP32 online-softmax accumulator, and K/V ring buffers.  At 61,312
tokens, the observed process peak falls from 7,108 MiB to 3,848 MiB while mean
latency also decreases.

The remaining dominant traffic is Q/K/V backing-store traffic.  If Q is divided
into `r` resident super-blocks, attention H2D remains:

```text
|Q| + r * (|K| + |V|)
```

Future optimization should therefore focus on maximizing useful resident Q
within the total model-memory budget, autotuning K/V tile size, and connecting
model-specific residual/gate epilogues to the output callback.  It should not
reintroduce a full raw-attention CPU tensor.
