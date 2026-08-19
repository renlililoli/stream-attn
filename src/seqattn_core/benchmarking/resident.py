from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch

from .common import ProcessMemorySampler, atomic_json, make_host_tensors_parallel

AttentionFunction = Callable[..., torch.Tensor]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_backend(name: str) -> tuple[AttentionFunction, dict[str, str]]:
    if name == "flash2":
        try:
            from flash_attn import flash_attn_func
        except ImportError as error:
            raise RuntimeError("flash2 requires the flash-attn package") from error
        return flash_attn_func, {
            "package": "flash-attn",
            "version": _package_version("flash-attn"),
            "function": "flash_attn.flash_attn_func",
        }
    if name == "flash4":
        try:
            from flash_attn.cute import flash_attn_func
        except ImportError as error:
            raise RuntimeError("flash4 requires the flash-attn-4 package") from error
        return flash_attn_func, {
            "package": "flash-attn-4",
            "version": _package_version("flash-attn-4"),
            "function": "flash_attn.cute.flash_attn_func",
        }
    raise ValueError(f"unsupported backend: {name}")


def _move_resident_tensor(host: torch.Tensor) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    device = host.unsqueeze(0).to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    residency_seconds = time.perf_counter() - started
    return device, residency_seconds


def _signature(output: torch.Tensor) -> dict[str, list[float]]:
    indices = sorted(
        {
            0,
            output.shape[1] // 4,
            output.shape[1] // 2,
            3 * output.shape[1] // 4,
            output.shape[1] - 1,
        }
    )
    return {
        str(index): output[0, index, 0, :8].float().cpu().tolist()
        for index in indices
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPU-resident attention backends")
    parser.add_argument("--backend", choices=("flash2", "flash4"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
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
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("resident benchmark requires a visible CUDA GPU")
        if args.repeats <= 0 or args.warmup < 0:
            raise ValueError("repeats must be positive and warmup must be non-negative")
        dtype = getattr(torch, args.dtype)
        attention, backend = _load_backend(args.backend)
        torch.set_num_threads(1)
        properties = torch.cuda.get_device_properties(0)
        result["environment"].update(
            gpu=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_gpu_memory_bytes=properties.total_memory,
        )
        result["backend"] = backend

        sampler = ProcessMemorySampler(args.sample_interval_ms / 1000)
        sampler.__enter__()
        started = time.perf_counter()
        q_host, k_host, v_host = make_host_tensors_parallel(
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
        data_seconds = time.perf_counter() - started
        residency_seconds = 0.0
        q, residency_part = _move_resident_tensor(q_host)
        residency_seconds += residency_part
        del q_host
        k, residency_part = _move_resident_tensor(k_host)
        residency_seconds += residency_part
        del k_host
        v, residency_part = _move_resident_tensor(v_host)
        residency_seconds += residency_part
        del v_host
        result.update(
            data_preparation_seconds=data_seconds,
            gpu_residency_preparation_seconds=residency_seconds,
            qkv_resident_bytes=q.nbytes + k.nbytes + v.nbytes,
        )

        scale = args.head_dim**-0.5

        def run_once() -> torch.Tensor:
            if args.backend == "flash4":
                result = attention(
                    q,
                    k,
                    v,
                    softmax_scale=scale,
                    causal=args.causal,
                )
                return result[0] if isinstance(result, tuple) else result
            return attention(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=args.causal,
            )

        output = None
        for _ in range(args.warmup):
            output = run_once()
            torch.cuda.synchronize()
            del output
            output = None
            torch.cuda.empty_cache()

        durations = []
        cuda_durations = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            if output is not None:
                del output
                torch.cuda.empty_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start_event.record()
            output = run_once()
            end_event.record()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            cuda_durations.append(start_event.elapsed_time(end_event) / 1000)
        sampler.__exit__()

        if output is None:
            raise RuntimeError("attention backend did not return an output tensor")
        output_signature = _signature(output)
        if not all(math.isfinite(value) for values in output_signature.values() for value in values):
            raise FloatingPointError("attention output contains NaN or Inf")
        mean_seconds = sum(durations) / len(durations)
        flop = 4 * args.q_heads * args.head_dim * args.tokens * args.tokens
        result.update(
            status="success",
            execution_seconds=durations,
            mean_execution_seconds=mean_seconds,
            cuda_event_seconds=cuda_durations,
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=flop / mean_seconds / 1e12,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            process_peak_rss_bytes=sampler.peak_rss_bytes,
            nvml_process_peak_bytes=sampler.peak_vram_bytes,
            memory_samples=sampler.samples,
            output_residency="gpu",
            output_signature=output_signature,
        )
    except Exception as error:  # noqa: BLE001 - benchmark records failures as JSON
        result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
