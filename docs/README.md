# SeqAttn documentation

This directory contains the current runtime contracts and the validation
reports that still inform supported configuration. Superseded design notes,
completed experiment plans, and obsolete benchmark snapshots are removed from
the working tree and remain available through Git history.

## Current documentation

| Topic | Document | Use it for |
|---|---|---|
| Runtime design | [`architecture.md`](architecture.md) | Package boundaries, execution families, memory ownership, and correctness invariants |
| Backend policy | [`backend_selection.md`](backend_selection.md) | Backend precedence, automatic selection, capability checks, and runtime restrictions |
| Q/KV calibration | [`q_chunk_calibration.md`](q_chunk_calibration.md) | Deployment-specific `q_chunk_tokens` and `kv_chunk_tokens` calibration |
| Activation estimation | [`activation_memory_estimation.md`](activation_memory_estimation.md) | Device-neutral memory/latency candidates and interactive allocation timelines |
| Paged and NVMe | [`paged_nvme_runtime.md`](paged_nvme_runtime.md) | Page contracts, host-memory budgets, storage modes, and direct-I/O behavior |
| H3 DiT runtime | [`design_dit_runtime.md`](design_dit_runtime.md) | Materialized and recomputed QKV policies, callbacks, and buffer ownership |
| H3 MLP tiling | [`design_dit_mlp_chunk_model.md`](design_dit_mlp_chunk_model.md) | Selecting projection and MLP tile sizes independently from attention chunks |

## Current validation

| Report | Scope |
|---|---|
| [`benchmark_h3_full_block_524k_2026-09-08.md`](benchmark_h3_full_block_524k_2026-09-08.md) | New live RTX 5090 524K full-block test, independent frozen prediction, warmups and timing error |
| [`benchmark_h3_full_estimator_2026-09-08.md`](benchmark_h3_full_estimator_2026-09-08.md) | Full H3 runner/callback model, actual CUDA checks and block-25 replay |
| [`benchmark_h3_block25_estimator_audit_2026-09-08.md`](benchmark_h3_block25_estimator_audit_2026-09-08.md) | Parent H3 block measurements, native stage timings and FFN carry mapping limitations |
| [`benchmark_activation_estimator_5090_2026-09-08.md`](benchmark_activation_estimator_5090_2026-09-08.md) | Offline estimator replay against retained RTX 5090 timing and allocated-memory measurements |
| [`benchmark_host_memory_roofline_2026-08.md`](benchmark_host_memory_roofline_2026-08.md) | Consolidated A30 and RTX 5090 host-memory roofline validation |
| [`benchmark_h3_qkv_recompute_profile_2026-08-27.md`](benchmark_h3_qkv_recompute_profile_2026-08-27.md) | MiniMax-H3 materialized versus QKV-recompute calibration |

Machine-readable observations remain under [`experiments/`](experiments/).
Figures remain under [`assets/`](assets/), and immutable release notes remain
under [`releases/`](releases/).

## Maintenance rules

- Put stable behavior in the current documents above.
- Name reusable validation reports `benchmark_<topic>_<date>.md`.
- Remove superseded documents from the working tree after their still-valid
  conclusions have been merged into a current document; Git history is the
  archive.
- Keep raw JSON as the source of truth for benchmark tables and figures.
- Update this index whenever a current document is added, replaced, or
  removed.
