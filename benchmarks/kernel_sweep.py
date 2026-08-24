from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from seqattn_core import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    build_plan,
)
from seqattn_core.benchmarking.common import atomic_json, make_bounds


@dataclass(frozen=True)
class KernelCase:
    name: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


CASES = (
    KernelCase("m64_n64_w4_s2", 64, 64, 4, 2),
    KernelCase("m32_n64_w4_s2", 32, 64, 4, 2),
    KernelCase("m128_n64_w4_s2", 128, 64, 4, 2),
    KernelCase("m64_n32_w4_s2", 64, 32, 4, 2),
    KernelCase("m64_n128_w4_s2", 64, 128, 4, 2),
    KernelCase("m64_n64_w2_s2", 64, 64, 2, 2),
    KernelCase("m64_n64_w8_s2", 64, 64, 8, 2),
    KernelCase("m64_n64_w4_s1", 64, 64, 4, 1),
    KernelCase("m64_n64_w4_s3", 64, 64, 4, 3),
    KernelCase("m64_n64_w4_s4", 64, 64, 4, 4),
    KernelCase("m128_n64_w8_s2", 128, 64, 8, 2),
    KernelCase("m64_n128_w8_s2", 64, 128, 8, 2),
    KernelCase("m128_n64_w8_s1", 128, 64, 8, 1),
    KernelCase("m128_n64_w8_s3", 128, 64, 8, 3),
    KernelCase("m128_n64_w2_s1", 128, 64, 2, 1),
    KernelCase("m128_n64_w2_s2", 128, 64, 2, 2),
    KernelCase("m128_n64_w2_s3", 128, 64, 2, 3),
    KernelCase("m128_n64_w4_s1", 128, 64, 4, 1),
    KernelCase("m128_n64_w4_s3", 128, 64, 4, 3),
    KernelCase("m128_n64_w4_s4", 128, 64, 4, 4),
    KernelCase("m128_n64_w8_s4", 128, 64, 8, 4),
    KernelCase("m64_n64_w8_s1", 64, 64, 8, 1),
    KernelCase("m64_n64_w8_s3", 64, 64, 8, 3),
    KernelCase("m64_n64_w8_s4", 64, 64, 8, 4),
    KernelCase("m128_n32_w4_s1", 128, 32, 4, 1),
    KernelCase("m128_n32_w4_s2", 128, 32, 4, 2),
    KernelCase("m128_n32_w8_s1", 128, 32, 8, 1),
    KernelCase("m128_n32_w8_s2", 128, 32, 8, 2),
    KernelCase("m128_n32_w8_s3", 128, 32, 8, 3),
    KernelCase("m128_n128_w8_s1", 128, 128, 8, 1),
    KernelCase("m128_n128_w8_s2", 128, 128, 8, 2),
    KernelCase("m64_n32_w8_s1", 64, 32, 8, 1),
    KernelCase("m64_n64_w4_s2_repeat", 64, 64, 4, 2),
)


def make_tensors_parallel(
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    *,
    seed: int,
    workers: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(torch.empty(shape, dtype=dtype, pin_memory=True) for shape in shapes)
    tasks = [
        (tensor_index, chunk_index, start, min(start + chunk_tokens, tensor.shape[0]))
        for tensor_index, tensor in enumerate(tensors)
        for chunk_index, start in enumerate(range(0, tensor.shape[0], chunk_tokens))
    ]

    def fill(task: tuple[int, int, int, int]) -> None:
        tensor_index, chunk_index, start, stop = task
        generator = torch.Generator(device="cpu").manual_seed(
            seed + tensor_index * 1_000_003 + chunk_index
        )
        tensors[tensor_index][start:stop].normal_(generator=generator)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seqattn-data") as pool:
        list(pool.map(fill, tasks))
    return tensors


def output_signature(output: torch.Tensor) -> torch.Tensor:
    tokens = output.shape[0]
    indices = sorted({0, tokens // 4, tokens // 2, 3 * tokens // 4, tokens - 1})
    return output[indices, 0, :8].float().clone()


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    delta = actual - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return {
        "sample_relative_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "sample_max_abs": float(delta.abs().max()),
        "sample_cosine": float(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0, eps=1e-12
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep seqattn Triton kernel launch parameters")
    parser.add_argument("--tokens", type=int, default=61_312)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--q-chunk", type=int, default=28_416)
    parser.add_argument("--kv-chunk", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", nargs="*", choices=tuple(case.name for case in CASES))
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
        "cases": [],
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        if args.warmup < 1 or args.repeats < 1:
            raise ValueError("warmup and repeats must be positive")
        torch.set_num_threads(1)
        dtype = getattr(torch, args.dtype)
        shape_q = (args.tokens, args.q_heads, args.head_dim)
        shape_kv = (args.tokens, args.kv_heads, args.head_dim)
        preparation_started = time.perf_counter()
        q, k, v = make_tensors_parallel(
            (shape_q, shape_kv, shape_kv),
            dtype,
            seed=args.seed,
            workers=args.cpu_workers,
            chunk_tokens=args.cpu_chunk_tokens,
        )
        output = torch.empty(shape_q, dtype=dtype, pin_memory=True)
        cu = make_bounds(args.tokens, args.segments)
        result["data_preparation_seconds"] = time.perf_counter() - preparation_started
        result["status"] = "running"
        atomic_json(args.output, result)

        reference_signature = None
        rows = result["cases"]
        assert isinstance(rows, list)
        selected_cases = (
            CASES
            if args.cases is None
            else tuple(case for case in CASES if case.name in args.cases)
        )
        for case in selected_cases:
            runner = plan = config = latest_stats = None
            try:
                config = StreamingAttentionConfig(
                    workspace_budget_bytes=args.workspace_mib * 2**20,
                    q_chunk_tokens=args.q_chunk,
                    kv_chunk_tokens=args.kv_chunk,
                    backend="triton",
                    block_m=case.block_m,
                    block_n=case.block_n,
                    num_warps=case.num_warps,
                    num_stages=case.num_stages,
                    num_kv_buffers=2,
                    num_output_buffers=1,
                    output_mode="host",
                )
                plan = build_plan(
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    device="cuda",
                    max_q_tokens=args.tokens,
                    max_kv_tokens=args.tokens,
                    config=config,
                )
                runner = StreamingAttentionRunner(plan, config)
                for _ in range(args.warmup):
                    runner(q, k, v, cu, cu, causal=args.causal, out=output)
                torch.cuda.synchronize()

                durations = []
                for _ in range(args.repeats):
                    stats = StreamingAttentionStats()
                    started = time.perf_counter()
                    runner(q, k, v, cu, cu, causal=args.causal, out=output, stats=stats)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                    latest_stats = stats
                signature = output_signature(output)
                if reference_signature is None:
                    reference_signature = signature
                mean_seconds = sum(durations) / len(durations)
                lengths = torch.diff(cu).tolist()
                flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in lengths)
                rows.append(
                    {
                        "status": "success",
                        **asdict(case),
                        "seconds": durations,
                        "mean_seconds": mean_seconds,
                        "effective_tflops": flop / mean_seconds / 1e12,
                        "plan": {
                            "q_chunk_tokens": plan.q_chunk_tokens,
                            "kv_chunk_tokens": plan.kv_chunk_tokens,
                            "estimated_workspace_bytes": plan.estimated_workspace_bytes,
                        },
                        "streaming_stats": (
                            latest_stats.as_dict() if latest_stats is not None else None
                        ),
                        **error_metrics(signature, reference_signature),
                    }
                )
            except Exception as error:  # noqa: BLE001 - preserve failed tuning points
                rows.append(
                    {
                        "status": "runtime_error",
                        **asdict(case),
                        "failure_message": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                atomic_json(args.output, result)
                del runner, plan, config, latest_stats
                gc.collect()
                torch.cuda.empty_cache()
        result["status"] = "success"
    except Exception as error:  # noqa: BLE001 - benchmark records failures as JSON
        result["status"] = (
            "oom" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
