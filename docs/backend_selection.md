# Backend selection and validation

Status: current for `seqattn-core 0.4.0a1` on 2026-09-02.

## Backend names

| Public value | Canonical backend | Availability |
|---|---|---|
| `builtin` or `triton` | `triton` | Included with the `cuda` or `dit` extra |
| `fa2`, `flash2`, or `flash2_split` | `fa2` | Optional `flash-attn` package |
| `fa3` | `fa3` | Optional FlashAttention-3 package |
| `fa4` | `fa4` | Optional `flash-attn-4` package |
| `reference` | `reference` | Always available on CPU |
| `auto` | architecture-aware selection | Uses the first compatible installed backend |

Aliases are accepted for compatibility, but statistics record canonical names.

## Algorithm mode versus backend

`StreamingAttentionConfig.backend` selects how one dense Q-by-K/V tile is
computed. It does not select a sparse attention algorithm.

MiniMax-H3 separately supports:

```toml
[minimax_h3]
attention_mode = "dense" # or "sol_streaming"
```

`sol_streaming` is an approximate routing algorithm built on SeqAttn's Triton
streaming runtime. It must not be passed as a backend name and is never selected
by `backend="auto"`. The H3 policy can deliberately keep configured early
denoising steps and blocks dense, but unsupported sparse contracts fail
explicitly rather than silently falling back.

## Configuration precedence

`StreamingAttentionConfig.backend=None` resolves while building the immutable
attention plan, in this order:

1. A non-`None` Python argument.
2. `SEQATTN_BACKEND`.
3. The TOML file named by `SEQATTN_CONFIG`.
4. `${XDG_CONFIG_HOME:-~/.config}/seqattn/config.toml` when it exists.
5. The built-in `auto` policy.

Passing `backend="auto"` explicitly bypasses environment and TOML selection.
An explicit backend fails when its package, dtype, device, head dimension, or
GPU capability is incompatible; it never silently changes to another backend.

```toml
[attention]
backend = "auto"

[minimax_h3]
execution_mode = "materialized"
projection_tile_tokens = 4096
ffn_tile_tokens = 4096
```

`SEQATTN_CONFIG` may point to a shared deployment file containing both tables.

## Automatic order

The current policy is implemented by
`seqattn_core.streaming.backend.automatic_backend_order`:

| Device | Preference order |
|---|---|
| CPU or unavailable CUDA | `reference` |
| SM80-SM89 | `fa2`, `triton`, `reference` |
| SM90-SM99 | `fa3`, `fa2`, `triton`, `reference` |
| SM100-SM109 | `fa4`, `triton`, `reference` |
| SM120+ | `triton`, `fa4`, `reference` |
| Other CUDA capability | `triton`, `reference` |

CUDA backends require FP16 or BF16. FlashAttention backends require head
dimension divisible by 8 and no greater than 256. FA3 requires SM90; FA4
requires Blackwell-class SM100 or newer.

## Runtime restrictions

| Runtime | Allowed backend behavior |
|---|---|
| Contiguous host-memory streaming | `triton`, `fa2`, `fa3`, `fa4`, or `reference` |
| Causal streaming with external offsets | Built-in Triton; automatic selection falls back, explicit Flash selection fails |
| Projected attention | Built-in Triton only |
| Recomputed attention | Built-in Triton only |
| H3 dense materialized or recompute runner | Built-in Triton only |
| H3 `sol_streaming` | Built-in Triton, BF16, head dimension 128, equal Q/KV heads, non-causal self-attention, SM80+, single GPU |
| Paged CUDA runtime | Built-in Triton only |
| Paged CPU runtime | `reference` only |

The Flash adapters produce a partial normalized output and FP32 LSE for one Q
tile and one K/V tile. SeqAttn owns the common streaming schedule and FP32
log-sum-exp combine path.

The Sol path adds one K/V summary prepass per packed segment. It still scans and
transfers every K/V tile for every resident Q chunk, and recompute mode must
project K/V once more for the summary. `effective_density` measures exact route
blocks divided by all routed blocks; it is not a transfer-density or end-to-end
speedup metric.

## Validation scope

Backend availability is a runtime capability check, not a performance claim.
Before adopting a backend for a deployment:

1. Run the relevant correctness tests for dtype, mask, head layout, and packed
   segment behavior.
2. Measure the resident backend and concurrent H2D roof on the final GPU and
   NUMA policy.
3. Calibrate Q chunk size using [`q_chunk_calibration.md`](q_chunk_calibration.md).
4. Keep independent process results and record package versions.

The consolidated A30 FA2 and RTX 5090 FA4/Triton validation is in
[`benchmark_host_memory_roofline_2026-08.md`](benchmark_host_memory_roofline_2026-08.md).

## Failure interpretation

- `unsupported backend` means the name is unknown.
- `requires ...` means an explicit optional package is not importable.
- GPU capability or dtype errors mean the package exists but cannot serve the
  requested plan.
- `no compatible seqattn backend is available` means every automatic candidate
  was unavailable or disallowed by the selected runtime.
- A `sol_streaming requires ...` error means the approximate H3 contract is
  incomplete or unsupported; the runtime does not substitute dense execution.
