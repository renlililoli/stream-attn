# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope and setup

`seqattn` is a standalone Linux/Python package for inference-only exact attention when Q/K/V exceed a fixed GPU or host-memory working set. It is also checked out as a submodule of the MiniMax-H3 integration repository; commits here and updates to the parent submodule pointer are separate operations.

Python 3.10+, PyTorch 2.5+, CUDA, and Triton 3.1+ are the supported runtime. The canonical integrated environment is the parent repository's Docker Compose service:

```bash
cd ../..
docker compose build diffsynth
docker compose up -d diffsynth
docker compose exec diffsynth bash
# Inside the container
cd /opt/seqattn
pip install -e '.[cuda,benchmark,dev]'
```

For a compatible standalone environment, install editable from this directory:

```bash
pip install -e '.[cuda,benchmark,dev]'
```

CPU reference, planner, storage, and API tests can run without CUDA. Triton tests skip when CUDA/Triton is unavailable, and direct-I/O tests may skip when the filesystem does not support `O_DIRECT`.

## Tests and quality checks

```bash
# Full suite
pytest -q

# One file or one test
pytest -q tests/test_planner.py
pytest -q tests/test_planner.py::test_planner_rejects_invalid_gqa_ratio

# Useful focused suites
pytest -q tests/test_reference.py
pytest -q tests/test_triton.py
pytest -q tests/test_pipeline.py
pytest -q tests/test_paged.py
pytest -q tests/test_paged_triton.py
pytest -q tests/test_nvme.py

# Lint and formatting verification
ruff check .
ruff format --check .

# Build distributions (requires the `build` package)
python -m build
```

Ruff targets Python 3.10 with a 100-column line length. `tests/test_module_layout.py` enforces the implementation/facade boundary described below.

## Package architecture

The source tree intentionally contains two packages:

- `src/seqattn_core/` owns all implementation. Add new logic here and import other internals through `seqattn_core` paths.
- `src/seqattn/` is a pure compatibility facade. Its modules only re-export `seqattn_core`; legacy paths such as `seqattn.runtime`, `seqattn.paging`, and `seqattn.pipeline` must preserve public object identity. Do not put implementation in this package or add compatibility-only modules to `seqattn_core`.

The three main execution paths share configuration, planning, validation, statistics, and Triton kernels:

1. **Contiguous streaming** (`streaming/`): CPU-backed packed Q/K/V are validated by `StreamingAttentionRunner`; `planner.py` jointly selects an HBM-resident Q super-block and streamed K/V tile; a persistent CUDA workspace overlaps H2D, Triton online-softmax updates, and D2H. `api.py` exposes dense and FlashAttention-style varlen wrappers.
2. **Paged memory/NVMe** (`paged/`, `storage/`): `PagedAttentionRunner` adapts `PageSource`/`PageSink` contracts, allocates all operator-owned host resources through `HostMemoryPlan`, streams Q and K/V pages through staging rings and a bounded two-region K/V cache, then dispatches to reference or Triton execution. `storage/` implements aligned Q/KV records, atomic manifest-last publication, output stores, and explicit `O_DIRECT` I/O.
3. **Projected pipeline** (`projection/`): `ProjectedAttentionRunner` pipelines CPU hidden-state H2D, model-owned QKV callbacks, and Q/K/V D2H into persistent pinned buffers. After a global K/V readiness barrier, streamed attention passes each GPU output tile directly to the model-owned output-projection callback before projected output D2H, avoiding a raw-attention CPU round trip.

`kernels/streaming.py` contains the fused Triton online-softmax update/finalization kernels. `benchmarking/` owns the installed CLI implementations; the top-level benchmark modules under `seqattn` are legacy entry-point facades.

## Execution and memory invariants

- The runtime is inference-only: backward, dropout, and arbitrary sparse masks are unsupported.
- `backend="auto"` selects Triton only for CUDA FP16/BF16 tensors when Triton is available; otherwise it uses the reference path. Explicit Triton also requires `head_dim >= 16`.
- Packed `cu_seqlens` are scheduler boundaries. Query and K/V tiles must never cross segments; causal attention uses bottom-right alignment for unequal Q/K lengths.
- Exact global self-attention cannot finalize a query until every key/value in its segment is ready; preserve the projected pipeline's global K/V barrier.
- Triton attention keeps FP32 running maximum, normalizer, and unnormalized output for only the resident Q super-block; it never materializes the score matrix.
- CUDA workspaces, streams, buffers, and events are persistent and single-flight. Reuse a compatible runner in repeated layers/steps; create separate runners for concurrent request streams.
- `workspace_budget_bytes` covers only `seqattn`-owned HBM, not CUDA context, weights, or caller allocations. The paged `host_memory_budget_bytes` likewise excludes tensors owned by `MemoryPageSource`/`MemoryPageSink` callers.
- Async Triton paths require pinned inputs/outputs unless explicitly configured otherwise. Passing a reusable pinned output avoids allocation cost and jitter.
- K/V cache hot slots are assigned from the descriptor order, not assumed to equal page IDs. Q bypasses the long-lived cache and outputs are sent to their sink as soon as staging permits.
- `direct_io=True` must fail explicitly when alignment/filesystem requirements are unmet; never silently fall back to buffered I/O. Store writers publish `manifest.json` only after data files are complete, validated, fsynced, and renamed.
- BF16/FP16 K/V storage is exact. INT8 K/V is an explicitly selected approximate mode with FP16 scales per 64 tokens/head; report its preparation time and numerical error separately.

## Benchmarks

Installed entry points are `seqattn-bench`, `seqattn-paged-bench`, and `seqattn-pipeline-bench`. Run comparison points in independent processes and treat emitted JSON as the result of record. Keep OOM and timeout outcomes; simulated NVMe is scheduling evidence, not physical-storage performance evidence.

```bash
seqattn-bench --mode seqattn --tokens 61312 \
  --q-heads 56 --kv-heads 56 --head-dim 128 \
  --workspace-mib 4096 --target-vram-mib 8192 \
  --kv-chunk 4096 --repeats 1 \
  --output benchmark-results/seqattn_61312.json

seqattn-pipeline-bench --mode pipeline --tokens 61312 \
  --hidden-size 5376 --heads 56 --head-dim 128 \
  --projection-chunk 2048 --workspace-mib 2048 --kv-chunk 4096 \
  --target-vram-mib 8192 --repeats 2 \
  --output benchmark-results/pipeline_61312.json

seqattn-paged-bench --storage simulated-nvme --tokens 61312 \
  --simulate-read-gbps 7 --simulate-write-gbps 6 \
  --simulate-read-latency-us 80 --simulate-write-latency-us 100 \
  --output benchmark-results/simulated_nvme_61312.json
```

Repository sweeps live in `benchmarks/`. `scripts/profile_nsys.sh` adds NVTX/Nsight tracing; profile timings are diagnostic and must not be reported as primary latency measurements. Physical-storage claims require a measured local device and `--formal-local-nvme`.
