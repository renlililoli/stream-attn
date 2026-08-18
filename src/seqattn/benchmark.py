from __future__ import annotations

import argparse
import json
import math
import os
import platform
import threading
import time
import traceback
from pathlib import Path

import torch

from . import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    build_plan,
)

try:
    import pynvml
except ImportError:  # pragma: no cover - optional benchmark dependency
    pynvml = None


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


_nvml_handle = None


def process_vram_mib(pid: int) -> float:
    global _nvml_handle
    if pynvml is None:
        return 0.0
    if _nvml_handle is None:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle)
    return sum(
        process.usedGpuMemory / 2**20
        for process in processes
        if process.pid == pid and process.usedGpuMemory is not None
    )


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_mib = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, process_vram_mib(self.pid))
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_mib = max(self.peak_mib, process_vram_mib(self.pid))


def make_bounds(tokens: int, segments: int) -> torch.Tensor:
    if segments <= 0 or segments > tokens:
        raise ValueError("segments must be within [1, tokens]")
    base, remainder = divmod(tokens, segments)
    lengths = [base + (index < remainder) for index in range(segments)]
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32)


def make_host_tensor(shape, dtype, generator):
    tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
    tensor.normal_(generator=generator)
    return tensor


def full_gpu_attention(mode, q_cpu, k_cpu, v_cpu, cu, causal, scale, out_cpu):
    q = q_cpu.to("cuda", non_blocking=True)
    k = k_cpu.to("cuda", non_blocking=True)
    v = v_cpu.to("cuda", non_blocking=True)
    if mode == "flash2":
        try:
            from flash_attn import flash_attn_func
        except ImportError as error:
            raise RuntimeError("flash2 benchmark requires flash-attn") from error
    for start, stop in zip(cu[:-1].tolist(), cu[1:].tolist()):
        q_tile = q[start:stop].unsqueeze(0)
        k_tile = k[start:stop].unsqueeze(0)
        v_tile = v[start:stop].unsqueeze(0)
        if mode == "flash2":
            tile = flash_attn_func(
                q_tile,
                k_tile,
                v_tile,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=causal,
            )
        else:
            q_sdpa = q_tile.transpose(1, 2)
            k_sdpa = k_tile.transpose(1, 2)
            v_sdpa = v_tile.transpose(1, 2)
            if q_sdpa.shape[1] != k_sdpa.shape[1]:
                group_size = q_sdpa.shape[1] // k_sdpa.shape[1]
                k_sdpa = k_sdpa.repeat_interleave(group_size, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(group_size, dim=1)
            tile = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=causal,
                scale=scale,
            ).transpose(1, 2)
        out_cpu[start:stop].copy_(tile.squeeze(0), non_blocking=True)
    torch.cuda.synchronize()
    return out_cpu


def configure_allocator(target_vram_mib: int | None, safety_mib: int) -> dict:
    torch.cuda.init()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    context_mib = process_vram_mib(os.getpid())
    metadata = {"context_mib": context_mib}
    if target_vram_mib is not None:
        allocator_limit_mib = target_vram_mib - context_mib - safety_mib
        if allocator_limit_mib <= 0:
            raise ValueError("target VRAM is smaller than CUDA context plus safety margin")
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(allocator_limit_mib / total_mib)
        metadata.update(
            target_vram_mib=target_vram_mib,
            allocator_limit_mib=allocator_limit_mib,
            safety_mib=safety_mib,
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU-backed sequence attention")
    parser.add_argument("--mode", choices=("seqattn", "flash2", "sdpa"), default="seqattn")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--q-chunk", type=int)
    parser.add_argument("--kv-chunk", type=int, default=4096)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--target-vram-mib", type=int)
    parser.add_argument("--safety-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "status": "runtime_error",
        "configuration": vars(args) | {"output": str(args.output)},
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    try:
        result["memory_policy"] = configure_allocator(args.target_vram_mib, args.safety_mib)
        dtype = getattr(torch, args.dtype)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        q = make_host_tensor(
            (args.tokens, args.q_heads, args.head_dim), dtype, generator
        )
        k = make_host_tensor(
            (args.tokens, args.kv_heads, args.head_dim), dtype, generator
        )
        v = make_host_tensor(
            (args.tokens, args.kv_heads, args.head_dim), dtype, generator
        )
        output_buffer = torch.empty(
            (args.tokens, args.q_heads, args.head_dim),
            dtype=dtype,
            pin_memory=True,
        )
        cu = make_bounds(args.tokens, args.segments)
        scale = args.head_dim**-0.5
        runner = None
        stats = None
        if args.mode == "seqattn":
            config = StreamingAttentionConfig(
                workspace_budget_bytes=args.workspace_mib * 2**20,
                q_chunk_tokens=args.q_chunk,
                kv_chunk_tokens=args.kv_chunk,
                backend="triton",
                enable_nvtx=args.nvtx,
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
            result["plan"] = {
                "q_chunk_tokens": plan.q_chunk_tokens,
                "kv_chunk_tokens": plan.kv_chunk_tokens,
                "estimated_workspace_mib": plan.estimated_workspace_bytes / 2**20,
            }

        def run_once():
            nonlocal stats
            if runner is not None:
                stats = StreamingAttentionStats()
                return runner(
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    causal=args.causal,
                    out=output_buffer,
                    stats=stats,
                )
            return full_gpu_attention(
                args.mode,
                q,
                k,
                v,
                cu,
                args.causal,
                scale,
                output_buffer,
            )

        for _ in range(args.warmup):
            run_once()
        torch.cuda.synchronize()
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = run_once()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        # NVML calls can perturb sub-second CUDA workloads.  Collect memory in
        # an untimed second pass so the primary latency remains uninstrumented.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with MemorySampler(args.sample_interval_ms / 1000) as sampler:
            probe_started = time.perf_counter()
            output = run_once()
            torch.cuda.synchronize()
            memory_probe_seconds = time.perf_counter() - probe_started
        if not torch.isfinite(output).all():
            raise FloatingPointError("attention output contains NaN or Inf")
        q_lengths = torch.diff(cu).tolist()
        flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in q_lengths)
        mean_seconds = sum(durations) / len(durations)
        result.update(
            status="success",
            seconds=durations,
            mean_seconds=mean_seconds,
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=flop / mean_seconds / 1e12,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            nvml_process_peak_mib=sampler.peak_mib,
            nvml_sample_count=sampler.samples,
            memory_probe_seconds=memory_probe_seconds,
        )
        if stats is not None:
            result["streaming_stats"] = stats.as_dict()
            result["logical_h2d_gib"] = stats.h2d_bytes / 2**30
            result["logical_d2h_gib"] = stats.d2h_bytes / 2**30
    except BaseException as error:
        result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
