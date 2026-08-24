from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch

from seqattn_core.benchmarking.common import atomic_json
from seqattn_core.kernels import update_attention_state


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_summary(seconds: list[float]) -> dict[str, float | int | list[float]]:
    return {
        "samples": len(seconds),
        "seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "p10_seconds": percentile(seconds, 0.10),
        "p90_seconds": percentile(seconds, 0.90),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
    }


def bandwidth_summary(
    seconds: list[float], payload_bytes: int
) -> dict[str, float | int | list[float]]:
    bytes_per_second = [payload_bytes / duration for duration in seconds]
    result = duration_summary(seconds)
    result.update(
        payload_bytes=payload_bytes,
        bytes_per_second=bytes_per_second,
        median_bytes_per_second=statistics.median(bytes_per_second),
        median_gb_per_second=statistics.median(bytes_per_second) / 1e9,
        median_gib_per_second=statistics.median(bytes_per_second) / 2**30,
        p10_gb_per_second=percentile(bytes_per_second, 0.10) / 1e9,
        p90_gb_per_second=percentile(bytes_per_second, 0.90) / 1e9,
    )
    return result


def make_event_pairs(count: int) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
    return [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(count)
    ]


def event_seconds(
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> list[float]:
    return [start.elapsed_time(end) / 1000.0 for start, end in pairs]


def capture_numa_artifacts(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    numa_maps = Path("/proc/self/numa_maps").read_text(encoding="ascii")
    (directory / "numa_maps.txt").write_text(numa_maps, encoding="ascii")
    page_counts: dict[str, int] = {}
    for line in numa_maps.splitlines():
        for field in line.split():
            if field.startswith("N") and "=" in field:
                node, value = field.split("=", 1)
                if node[1:].isdigit() and value.isdigit():
                    page_counts[node] = page_counts.get(node, 0) + int(value)
    numastat = subprocess.run(
        ["numastat", "-p", str(os.getpid())],
        check=False,
        capture_output=True,
        text=True,
    )
    (directory / "numastat.txt").write_text(numastat.stdout, encoding="ascii")
    if numastat.stderr:
        (directory / "numastat.stderr.txt").write_text(numastat.stderr, encoding="ascii")
    return {
        "numa_maps_page_counts": page_counts,
        "numastat_returncode": numastat.returncode,
        "numa_artifact_directory": str(directory),
    }


def run_bare(
    k_host: torch.Tensor,
    v_host: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    k_buffers = [torch.empty_like(k_host, device="cuda") for _ in range(2)]
    v_buffers = [torch.empty_like(v_host, device="cuda") for _ in range(2)]
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        for index in range(warmup):
            slot = index % 2
            k_buffers[slot].copy_(k_host, non_blocking=True)
            v_buffers[slot].copy_(v_host, non_blocking=True)
    stream.synchronize()

    pairs = make_event_pairs(repeats)
    with torch.cuda.stream(stream):
        for index, (start, end) in enumerate(pairs):
            slot = index % 2
            start.record(stream)
            k_buffers[slot].copy_(k_host, non_blocking=True)
            v_buffers[slot].copy_(v_host, non_blocking=True)
            end.record(stream)
    stream.synchronize()

    payload_bytes = k_host.nbytes + v_host.nbytes
    return bandwidth_summary(event_seconds(pairs), payload_bytes)


def launch_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    accumulator: torch.Tensor,
    *,
    initialize: bool,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> None:
    update_attention_state(
        q,
        k,
        v,
        running_max,
        running_sum,
        accumulator,
        q_tokens=q.shape[0],
        kv_tokens=k.shape[0],
        q_local_offset=0,
        kv_local_offset=0,
        causal_shift=0,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        initialize=initialize,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_concurrent(
    k_host: torch.Tensor,
    v_host: torch.Tensor,
    *,
    q_tokens: int,
    q_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + q_tokens)
    q = torch.randn(
        (q_tokens, q_heads, head_dim),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    k_buffers = [torch.empty_like(k_host, device="cuda") for _ in range(2)]
    v_buffers = [torch.empty_like(v_host, device="cuda") for _ in range(2)]
    running_max = torch.empty((q_tokens, q_heads), dtype=torch.float32, device="cuda")
    running_sum = torch.empty_like(running_max)
    accumulator = torch.empty((q_tokens, q_heads, head_dim), dtype=torch.float32, device="cuda")
    compute_stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream()
    kv_ready = [torch.cuda.Event() for _ in range(2)]
    kv_free = [torch.cuda.Event() for _ in range(2)]

    with torch.cuda.stream(h2d_stream):
        for slot in range(2):
            k_buffers[slot].copy_(k_host, non_blocking=True)
            v_buffers[slot].copy_(v_host, non_blocking=True)
            kv_ready[slot].record(h2d_stream)
    h2d_stream.synchronize()

    with torch.cuda.stream(compute_stream):
        compute_stream.wait_event(kv_ready[0])
        launch_update(
            q,
            k_buffers[0],
            v_buffers[0],
            running_max,
            running_sum,
            accumulator,
            initialize=True,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        kv_free[0].record(compute_stream)
    compute_stream.synchronize()

    def enqueue_iteration(
        index: int,
        copy_pair: tuple[torch.cuda.Event, torch.cuda.Event] | None,
        compute_pair: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    ) -> None:
        current = index % 2
        target = 1 - current
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(kv_ready[current])
            if compute_pair is not None:
                compute_pair[0].record(compute_stream)
            launch_update(
                q,
                k_buffers[current],
                v_buffers[current],
                running_max,
                running_sum,
                accumulator,
                initialize=False,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            if compute_pair is not None:
                compute_pair[1].record(compute_stream)
            kv_free[current].record(compute_stream)
        with torch.cuda.stream(h2d_stream):
            if index > 0:
                h2d_stream.wait_event(kv_free[target])
            if copy_pair is not None:
                copy_pair[0].record(h2d_stream)
            k_buffers[target].copy_(k_host, non_blocking=True)
            v_buffers[target].copy_(v_host, non_blocking=True)
            if copy_pair is not None:
                copy_pair[1].record(h2d_stream)
            kv_ready[target].record(h2d_stream)

    for index in range(warmup):
        enqueue_iteration(index, None, None)
    compute_stream.synchronize()
    h2d_stream.synchronize()

    copy_pairs = make_event_pairs(repeats)
    compute_pairs = make_event_pairs(repeats)
    for index in range(repeats):
        enqueue_iteration(index + warmup, copy_pairs[index], compute_pairs[index])
    compute_stream.synchronize()
    h2d_stream.synchronize()

    payload_bytes = k_host.nbytes + v_host.nbytes
    return {
        "copy": bandwidth_summary(event_seconds(copy_pairs), payload_bytes),
        "compute": duration_summary(event_seconds(compute_pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate bare and compute-concurrent pinned H2D K/V bandwidth"
    )
    parser.add_argument("--mode", choices=("bare", "concurrent", "both"), default="both")
    parser.add_argument("--kv-chunk", type=int, default=4096)
    parser.add_argument("--compute-q", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128), default=128)
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64, 128), default=64)
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--numa-artifact-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }

    result: dict[str, object] = {
        "status": "runtime_error",
        "configuration": configuration,
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
            raise RuntimeError("a visible CUDA GPU is required")
        if args.warmup < 0 or args.repeats <= 0:
            raise ValueError("warmup must be non-negative and repeats must be positive")
        if args.kv_chunk <= 0 or any(q <= 0 for q in args.compute_q):
            raise ValueError("K/V and Q chunk sizes must be positive")
        if args.q_heads <= 0 or args.kv_heads <= 0 or args.q_heads % args.kv_heads:
            raise ValueError("q_heads must be a positive multiple of kv_heads")
        if any(q % args.block_m for q in args.compute_q):
            raise ValueError("every compute Q chunk must be divisible by block_m")
        if args.kv_chunk % args.block_n:
            raise ValueError("kv_chunk must be divisible by block_n")

        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        properties = torch.cuda.get_device_properties(0)
        result["environment"].update(
            gpu=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_gpu_memory_bytes=properties.total_memory,
        )

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        shape = (args.kv_chunk, args.kv_heads, args.head_dim)
        k_host = torch.empty(shape, dtype=dtype, pin_memory=True)
        v_host = torch.empty(shape, dtype=dtype, pin_memory=True)
        k_host.normal_(generator=generator)
        v_host.normal_(generator=generator)
        result.update(
            status="allocated",
            tensor_shape=list(shape),
            k_bytes=k_host.nbytes,
            v_bytes=v_host.nbytes,
            kv_payload_bytes=k_host.nbytes + v_host.nbytes,
            host_tensors_pinned=k_host.is_pinned() and v_host.is_pinned(),
        )
        if args.numa_artifact_dir is not None:
            result["numa"] = capture_numa_artifacts(args.numa_artifact_dir)
        atomic_json(args.output, result)
        print(
            f"ALLOCATED pid={os.getpid()} payload_bytes={k_host.nbytes + v_host.nbytes}",
            flush=True,
        )
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)

        if args.mode in {"bare", "both"}:
            result["bare"] = run_bare(
                k_host,
                v_host,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            atomic_json(args.output, result)

        if args.mode in {"concurrent", "both"}:
            concurrent: dict[str, object] = {}
            for q_tokens in args.compute_q:
                concurrent[str(q_tokens)] = run_concurrent(
                    k_host,
                    v_host,
                    q_tokens=q_tokens,
                    q_heads=args.q_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    block_m=args.block_m,
                    block_n=args.block_n,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    seed=args.seed,
                )
                result["concurrent"] = concurrent
                atomic_json(args.output, result)

        result["status"] = "success"
    except Exception as error:  # noqa: BLE001 - benchmark must publish failures
        result["status"] = (
            "oom" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
