# Backend selection and validation

## Runtime paths

The contiguous CPU-DRAM streaming runner has one shared H2D/compute/D2H
schedule and one shared FP32 log-sum-exp combine implementation. The backend
only supplies the partial attention result for one resident Q tile and one K/V
tile:

```text
partial_forward(Q, K_tile, V_tile) -> normalized output, FP32 LSE
```

The adapter registry contains four CUDA paths:

| Backend | Implementation | Availability behavior |
|---|---|---|
| `builtin` | SeqAttn Triton update/finalize kernels | Included with the Triton extra |
| `fa2` | FlashAttention 2 low-level forward with preallocated output | Optional `flash-attn` package |
| `fa3` | FlashAttention 3 public forward returning output and LSE | Optional FA3 package |
| `fa4` | FlashAttention 4 CuTe public forward returning output and LSE | Optional `flash-attn-4` package |

The historical names `triton`, `flash2`, and `flash2_split` remain accepted as
aliases. Statistics record the canonical implementation name (`triton`,
`fa2`, `fa3`, or `fa4`).

FA2, FA3, and FA4 differ at their Python boundary, so each has a small explicit
adapter. The streaming executor does not inspect signatures or contain
version-specific branches. Each adapter normalizes LSE to contiguous FP32
`[batch, heads, query_tokens]` before the shared combine kernel runs.

## Automatic policy

`StreamingAttentionConfig.backend=None` enables external configuration. The
precedence is:

1. A non-`None` Python `backend` argument.
2. `SEQATTN_BACKEND`.
3. The TOML file selected by `SEQATTN_CONFIG`.
4. `${XDG_CONFIG_HOME:-~/.config}/seqattn/config.toml` when present.
5. The built-in `auto` policy.

The TOML schema is intentionally small:

```toml
[attention]
backend = "auto"
```

Supported values are `auto`, `builtin`, `fa2`, `fa3`, `fa4`, and `reference`.
Explicit selection fails when the requested dependency or GPU architecture is
not available. Automatic selection falls through its architecture-specific
order:

| Compute capability | Preference order |
|---|---|
| SM80-SM89 | `fa2`, `builtin`, `reference` |
| SM90-SM99 | `fa3`, `fa2`, `builtin`, `reference` |
| SM100-SM109 | `fa4`, `builtin`, `reference` |
| SM120+ | `builtin`, `fa4`, `reference` |

The paged and projected-output runtimes currently restrict their selection to
the built-in kernel and reference implementations because their staging and
device-consumer contracts are not implemented by the FA adapters.

## Validated hardware

### NVIDIA A30 / SM80

The `fa2` streaming backend was validated on August 24, 2026 in the
`seqattn-a30` environment with FlashAttention `2.7.4.post1`, BF16, 56 heads,
head dimension 128, 409,600 tokens, an 8,192-token K/V chunk, and a 2GiB
workspace. Host Q/K/V generation used 32 CPU workers and was excluded from the
execution timing.

| Path | Execution | Effective TFLOPS |
|---|---:|---:|
| Resident FA2 | 50.9142 s | 94.48 |
| SeqAttn `fa2`, 2GiB | 51.1511 s | 94.04 |
| SeqAttn built-in, 2GiB | 59.6702 s | 80.62 |

The streamed FA2 path was 0.47% slower than resident FA2 and 16.65% faster than
the built-in kernel for this shape. Its sampled output had relative L2 error
0.00456 and cosine similarity 0.9999909 against resident FA2.

### NVIDIA RTX 5090 / SM120

The built-in kernel remains the validated RTX 5090 path. The August 19, 2026
524,288-token BF16 measurement used the unchanged SM120 launch profile
`128x64`, 8 warps, and 3 stages. At a 2GiB workspace it measured 36.247 seconds
and 217.4 effective TFLOPS, within 0.06% of the separately measured resident
FA4 execution time.

The FA4 streaming adapter is available for explicit selection, but it has not
yet received the same end-to-end streaming validation in this branch. Keeping
`builtin` first for SM120 prevents package installation from changing the
validated default.

### FA3 and FA4 validation status

FA3 and FA4 adapter contracts are covered by unit tests with their documented
output/LSE layouts. Hardware execution still needs to be run in matching SM90
and SM100/SM120 environments before those paths are marked production-
validated.
