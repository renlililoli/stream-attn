from __future__ import annotations

import argparse
import json
import math
import platform
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..config import StreamingAttentionConfig
from ..planner import build_plan
from ..stats import StreamingAttentionStats
from ..streaming import StreamingAttentionRunner
from .common import (
    MemorySampler,
    atomic_json,
    configure_allocator,
    make_bounds,
    make_host_tensors_parallel,
    make_pinned_host_tensors_parallel,
)


def output_signature(output: torch.Tensor) -> dict[str, list[float]]:
    indices = sorted(
        {
            0,
            output.shape[0] // 4,
            output.shape[0] // 2,
            3 * output.shape[0] // 4,
            output.shape[0] - 1,
        }
    )
    return {
        str(index): output[index, 0, :8].float().tolist()
        for index in indices
    }


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
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8))
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--num-kv-buffers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--num-output-buffers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--target-vram-mib", type=int)
    parser.add_argument("--safety-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=20)
    parser.add_argument("--skip-memory-probe", action="store_true")
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
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
        preparation_started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="seqattn-prepare"
        ) as pool:
            inputs = pool.submit(
                make_host_tensors_parallel,
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
            output = pool.submit(
                make_pinned_host_tensors_parallel,
                ((args.tokens, args.q_heads, args.head_dim),),
                dtype,
                workers=1,
            )
            q, k, v = inputs.result()
            (output_buffer,) = output.result()
        result["data_preparation_seconds"] = time.perf_counter() - preparation_started
        cu = make_bounds(args.tokens, args.segments)
        scale = args.head_dim**-0.5
        runner = None
        stats = None
        if args.mode == "seqattn":
            config = StreamingAttentionConfig(
                workspace_budget_bytes=args.workspace_mib * 2**20,
                q_chunk_tokens=args.q_chunk,
                kv_chunk_tokens=args.kv_chunk,
                block_m=args.block_m,
                block_n=args.block_n,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                num_kv_buffers=args.num_kv_buffers,
                num_output_buffers=args.num_output_buffers,
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
                "q_chunk_requested_tokens": args.q_chunk,
                "q_passes": math.ceil(args.tokens / plan.q_chunk_tokens),
                "q_effective_tokens": args.tokens
                / math.ceil(args.tokens / plan.q_chunk_tokens),
                "kv_chunk_tokens": plan.kv_chunk_tokens,
                "block_m": plan.block_m,
                "block_n": plan.block_n,
                "num_warps": plan.num_warps,
                "num_stages": plan.num_stages,
                "num_kv_buffers": plan.num_kv_buffers,
                "num_output_buffers": plan.num_output_buffers,
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
        compute_pipeline_durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = run_once()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            if stats is not None:
                compute_pipeline_durations.append(stats.compute_pipeline_seconds)
        # NVML calls can perturb CUDA workloads. Collect memory in an untimed
        # second pass unless a sweep explicitly opts out to avoid one full
        # extra long-sequence execution per point.
        memory_probe_seconds = None
        nvml_process_peak_mib = None
        nvml_sample_count = 0
        if not args.skip_memory_probe:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with MemorySampler(args.sample_interval_ms / 1000) as sampler:
                probe_started = time.perf_counter()
                output = run_once()
                torch.cuda.synchronize()
                memory_probe_seconds = time.perf_counter() - probe_started
            nvml_process_peak_mib = sampler.peak_mib
            nvml_sample_count = sampler.samples
        if not torch.isfinite(output).all():
            raise FloatingPointError("attention output contains NaN or Inf")
        q_lengths = torch.diff(cu).tolist()
        flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in q_lengths)
        mean_seconds = sum(durations) / len(durations)
        mean_compute_pipeline_seconds = (
            sum(compute_pipeline_durations) / len(compute_pipeline_durations)
            if compute_pipeline_durations
            else None
        )
        result.update(
            status="success",
            seconds=durations,
            mean_seconds=mean_seconds,
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=flop / mean_seconds / 1e12,
            compute_pipeline_seconds=compute_pipeline_durations,
            mean_compute_pipeline_seconds=mean_compute_pipeline_seconds,
            compute_pipeline_effective_tflops=(
                flop / mean_compute_pipeline_seconds / 1e12
                if mean_compute_pipeline_seconds is not None
                else None
            ),
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            nvml_process_peak_mib=nvml_process_peak_mib,
            nvml_sample_count=nvml_sample_count,
            memory_probe_seconds=memory_probe_seconds,
            memory_probe_skipped=args.skip_memory_probe,
            output_signature=output_signature(output),
        )
        if stats is not None:
            result["streaming_stats"] = stats.as_dict()
            element_size = torch.empty((), dtype=dtype).element_size()
            q_h2d_bytes = args.tokens * args.q_heads * args.head_dim * element_size
            kv_h2d_bytes = stats.h2d_bytes - q_h2d_bytes
            result["one_time_q_h2d_bytes"] = q_h2d_bytes
            result["logical_kv_h2d_bytes"] = kv_h2d_bytes
            result["logical_d2h_bytes"] = stats.d2h_bytes
            result["logical_h2d_gib"] = stats.h2d_bytes / 2**30
            result["logical_kv_h2d_gib"] = kv_h2d_bytes / 2**30
            result["logical_d2h_gib"] = stats.d2h_bytes / 2**30
    except Exception as error:  # noqa: BLE001 - benchmark records failures as JSON
        result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
