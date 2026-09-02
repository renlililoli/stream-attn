from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

import torch
from large_tier_comparison import (
    OutputSignature,
    activation_sizes,
    make_tensors_parallel,
)

from seqattn_core import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    build_attention_plan,
)
from seqattn_core.benchmarking.common import ProcessMemorySampler, make_bounds


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep unconstrained DRAM-backed HBM workspaces")
    parser.add_argument("--tokens", type=int, default=524_288)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--workspace-gib", nargs="+", type=float, default=[4, 6, 8, 10, 12, 16])
    parser.add_argument("--kv-chunk", type=int, default=8192)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8))
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "status": "runtime_error",
        "configuration": vars(args) | {"output": str(args.output)},
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
        },
        "storage_backend": "dram",
        "storage_performance_valid": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        if args.cpu_workers <= 0 or args.cpu_chunk_tokens <= 0:
            raise ValueError("CPU generation worker and chunk counts must be positive")
        torch.set_num_threads(1)
        dtype = getattr(torch, args.dtype)
        sizes = activation_sizes(args.tokens, args.q_heads, args.kv_heads, args.head_dim, dtype)
        result["activation_sizes"] = sizes

        preparation_started = time.perf_counter()
        q, k, v = make_tensors_parallel(
            (
                (args.tokens, args.q_heads, args.head_dim),
                (args.tokens, args.kv_heads, args.head_dim),
                (args.tokens, args.kv_heads, args.head_dim),
            ),
            dtype,
            seed=args.seed,
            workers=args.cpu_workers,
            chunk_tokens=args.cpu_chunk_tokens,
        )
        cu = make_bounds(args.tokens, args.segments)
        output = torch.empty(q.shape, dtype=q.dtype, pin_memory=True)
        preparation_seconds = time.perf_counter() - preparation_started
        result["data_preparation_seconds"] = preparation_seconds
        result["data_preparation_gib_per_second"] = sizes["qkv_bytes"] / 2**30 / preparation_seconds
        rows = []
        result.update(status="running", rows=rows)
        atomic_json(args.output, result)
        for workspace_gib in args.workspace_gib:
            workspace_bytes = int(workspace_gib * 2**30)
            config = StreamingAttentionConfig(
                workspace_budget_bytes=workspace_bytes,
                kv_chunk_tokens=args.kv_chunk,
                backend="triton",
                block_m=args.block_m,
                block_n=args.block_n,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                num_kv_buffers=2,
                num_output_buffers=1,
                output_mode="host",
            )
            plan = build_attention_plan(
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device="cuda",
                max_q_tokens=args.tokens,
                max_kv_tokens=args.tokens,
                config=config,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            signature = OutputSignature(args.tokens)
            stats = StreamingAttentionStats()
            runner = StreamingAttentionRunner(plan)
            with ProcessMemorySampler(args.sample_interval_ms / 1000) as sampler:
                torch.cuda.synchronize()
                started = time.perf_counter()
                runner(
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    causal=args.causal,
                    out=output,
                    stats=stats,
                )
                torch.cuda.synchronize()
                execution_seconds = time.perf_counter() - started
            signature.from_cpu(output)
            lengths = torch.diff(cu).tolist()
            flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in lengths)
            rows.append(
                {
                    "status": "success",
                    "workspace_gib": workspace_gib,
                    "execution_seconds": execution_seconds,
                    "tokens_per_second": args.tokens / execution_seconds,
                    "effective_tflops": flop / execution_seconds / 1e12,
                    "q_chunk_tokens": plan.q_chunk_tokens,
                    "kv_chunk_tokens": plan.kv_chunk_tokens,
                    "block_m": plan.block_m,
                    "block_n": plan.block_n,
                    "num_warps": plan.num_warps,
                    "num_stages": plan.num_stages,
                    "estimated_workspace_bytes": plan.estimated_workspace_bytes,
                    "process_peak_rss_bytes": sampler.peak_rss_bytes,
                    "nvml_process_peak_bytes": sampler.peak_vram_bytes,
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "memory_samples": sampler.samples,
                    "output_signature": signature.as_dict(),
                    "streaming_stats": stats.as_dict(),
                }
            )
            atomic_json(args.output, result)
            del runner, plan, config, stats
            torch.cuda.empty_cache()
        result["status"] = "success"
    except Exception as error:  # noqa: BLE001 - benchmark must publish failures as JSON
        result["status"] = (
            "oom" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
