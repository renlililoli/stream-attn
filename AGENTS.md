# SeqAttn Agent Guide

This repository is the independent Apache-2.0 `seqattn-core` package. Keep it
usable without ComfyUI. Framework adapters consume this package through public
callbacks and immutable release commits.

## Scope

Edit this repository for:

- exact attention planning and execution;
- Triton and FlashAttention split backends;
- projected hidden-to-QKV pipelines;
- paged host-memory and NVMe storage;
- shared H3 DiT callback types and tile configuration;
- operator benchmarks and hardware calibration guides.

Do not import `comfy`, `comfy_aimdo`, ComfyUI node classes, Dynamic VBAR
objects, or checkpoint loaders. ComfyUI weight preparation, block eviction,
Qwen/VAE patches, workflows, and UI controls belong in the community adapter.

## Start Safely

```bash
git status --short --branch
git branch -vv
git tag --sort=-version:refname
```

Do not discard untracked experiment notes or generated evidence. Work on a
feature branch, then integrate through a clean `main` worktree for release.

As of August 28, 2026, `main` is preparing `v0.3.0-alpha.4`; verify the remote
tag, GitHub prerelease, and immutable commit before updating a consumer pin.

## Package Boundaries

Core code lives under `src/seqattn_core`. Optional multi-GPU code lives in the
separate `packages/seqattn-multigpu` distribution and consumes the versioned
private plugin bridge. Public core names are exported from
`src/seqattn_core/__init__.py`; core must not export `MultiGpu*` symbols.

The main execution families are:

- `streaming/`: contiguous pinned-host Q/K/V with resident-Q and streamed-K/V;
- `projection/`: hidden H2D, consumer QKV projection, global K/V barrier,
  attention, and consumer output projection;
- `paged/` and `storage/`: bounded host cache and physical/simulated storage;
- `dit/`: generic H3 block orchestration through `H3BlockOps` callbacks;
- `kernels/`: Triton kernels and launch profiles.

Keep H3 integration generic. `H3MaterializedRunner` and `H3RecomputeRunner` may
schedule callback operations, but the consumer owns model-specific normalization,
projection, residual, MLP, and weight lifecycle behavior.

## Correctness and Memory Invariants

- The runtime is inference-only unless a new training contract is explicitly
  designed and tested.
- Packed `cu_seqlens` are hard segment boundaries. No Q or K/V tile may cross a
  segment.
- Exact dense attention requires all K/V for a segment before finalizing Q.
- Online softmax state remains FP32 and bounded by the resident Q super-block.
- Persistent workspaces, streams, buffers, and events are single-flight. Reuse
  a runner for compatible repeated calls and create separate runners for true
  concurrency.
- `workspace_budget_bytes` covers only SeqAttn-owned CUDA buffers, not model
  weights, CUDA context, caller tensors, or the whole process.
- Host-memory budgets exclude caller-owned sources/sinks unless the API says
  otherwise.
- Pinned inputs are required by asynchronous CUDA paths.
- `direct_io=True` fails explicitly when filesystem or alignment requirements
  are unmet. Never silently fall back to buffered I/O.
- Exact BF16/FP16 storage and approximate INT8 storage must remain separately
  named and measured.

The community package does not expose a user-facing attention workspace knob,
but this generic core API remains valid for standalone and paged execution.

## Configuration

Backend selection follows the documented explicit argument/environment/TOML
precedence. Model execution and stage tiling use one naming scheme:

```toml
[minimax_h3]
execution_mode = "materialized" # or "recompute"
projection_tile_tokens = 4096
ffn_tile_tokens = 4096

[wan]
execution_mode = "materialized" # or "recompute"
projection_tile_tokens = 2048
ffn_tile_tokens = 2048

[ltx2]
execution_mode = "materialized" # or "recompute"
projection_tile_tokens = 2048
video_ffn_tile_tokens = 2048
audio_ffn_tile_tokens = 2048
```

`SEQATTN_CONFIG` may point to a shared deployment file. Do not put ComfyUI-only
environment variables into the core configuration. H3, Wan, and LTX2 use the
same execution-mode and projection-tile naming rules. Wan and LTX2 defaults are
conservative starting values, not calibrated performance defaults.

`q_chunk_tokens` is normally calibrated from measured host-memory bandwidth
and resident attention throughput. Do not select it from nominal PCIe bandwidth
or advertised GPU TFLOPS. Follow `docs/q_chunk_calibration.md` and preserve the
GPU, backend, CPU affinity, NUMA policy, and memory-population details.

## Development and Tests

Install a development environment with:

```bash
pip install -e '.[cuda,dit,dev]'
```

Use focused tests while developing:

```bash
pytest -q tests/test_planner.py
pytest -q tests/test_reference.py
pytest -q tests/test_triton.py
pytest -q tests/test_pipeline.py
pytest -q tests/test_dit.py
pytest -q tests/test_paged.py tests/test_paged_triton.py
pytest -q tests/test_nvme.py
```

Before release, run:

```bash
pytest -q
ruff check .
ruff format --check .
python -m build
```

CPU-only success does not validate Triton, FlashAttention, asynchronous copy,
or direct-I/O paths. Run the relevant CUDA tests in the canonical GPU image.
Direct-I/O skips are acceptable only when the filesystem genuinely lacks the
required support and the skipped path is not part of the release claim.

## Benchmark Discipline

- Run comparison points in independent processes.
- Treat emitted JSON as the source of truth and retain failures/timeouts.
- Separate warmup/compile from steady measurements.
- Do not use Nsight timings as primary latency numbers.
- Simulated NVMe demonstrates scheduler behavior, not physical storage speed.
- Physical storage claims require a measured device and formal local-NVMe mode.
- A sweep-selected Q chunk or launch profile requires an independent idle-GPU
  winner retest before it becomes a default or README claim.
- Do not mix results from different backends, NUMA policies, memory population,
  or revisions in one comparison table without labeling them explicitly.

## Commit Discipline

- Keep implementation, benchmark evidence, and release metadata changes easy
  to review.
- Do not commit caches, accidental build products, or unrelated experiment
  output.
- Do not update a consumer's pin from this repository; make the core commit and
  release first, then update consumers separately.
- Do not move published tags or rewrite release history.

## Release Procedure

SeqAttn releases are GitHub source releases consumed through immutable commit
archives. There is currently no PyPI release step or automated release
workflow, so an agent must perform and verify each step explicitly.

1. Start from a clean `main` worktree and integrate the reviewed feature
   commits.
2. Update the PEP 440 version in `pyproject.toml`, for example `0.3.0a4`.
3. Add `docs/releases/v0.3.0-alpha.4.md` with user-visible changes, compatibility
   notes, and validated configuration.
4. Run the complete release checks and the CUDA tests/benchmarks required by
   the changed paths.
5. Commit and push `main`.
6. Create an annotated public tag at that exact commit:

   ```bash
   git tag -a v0.3.0-alpha.4 -m 'seqattn-core v0.3.0-alpha.4'
   git push origin main
   git push origin v0.3.0-alpha.4
   ```

7. Create a GitHub prerelease from the checked-in notes:

   ```bash
   gh release create v0.3.0-alpha.4 \
     --repo renlililoli/stream-attn \
     --verify-tag --prerelease \
     --title 'seqattn-core v0.3.0-alpha.4' \
     --notes-file docs/releases/v0.3.0-alpha.4.md
   ```

8. Verify the tag object resolves to the tested commit, the Release is not a
   draft, and a clean install from the commit archive works with `[dit]`.
9. Give consumers the immutable release commit SHA. Community metadata,
   Dockerfiles, notices, tests, and docs must all pin the same SHA.

Do not describe a tag-only state as released. Release completion requires the
remote tag, published GitHub prerelease, install verification, and recorded
commit SHA.
