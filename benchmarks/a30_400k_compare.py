from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from seqattn import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    StreamingAttentionStats,
    build_plan,
)
from seqattn.benchmarking.common import atomic_json, make_bounds, make_host_tensors_parallel


def output_signature(output: torch.Tensor) -> dict[str, list[float]]:
    tokens = output.shape[-3]
    indices = sorted({0, tokens // 4, tokens // 2, 3 * tokens // 4, tokens - 1})
    if output.device.type == "cuda":
        return {
            str(index): output[0, index, 0, :8].float().cpu().tolist()
            for index in indices
        }
    return {str(index): output[index, 0, :8].float().tolist() for index in indices}


def run_resident_fa2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    flop: int,
) -> dict[str, object]:
    from flash_attn import flash_attn_func

    residency_started = time.perf_counter()
    q_gpu = q.unsqueeze(0).to("cuda", non_blocking=True)
    k_gpu = k.unsqueeze(0).to("cuda", non_blocking=True)
    v_gpu = v.unsqueeze(0).to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    residency_seconds = time.perf_counter() - residency_started

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    output = flash_attn_func(
        q_gpu,
        k_gpu,
        v_gpu,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )
    end_event.record()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "status": "success",
        "seconds": seconds,
        "cuda_seconds": start_event.elapsed_time(end_event) / 1000.0,
        "effective_tflops": flop / seconds / 1e12,
        "gpu_residency_preparation_seconds": residency_seconds,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "output_signature": output_signature(output),
    }
    del output, q_gpu, k_gpu, v_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_streaming(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    output: torch.Tensor,
    *,
    backend: str,
    workspace_mib: int,
    kv_chunk: int,
    flop: int,
) -> dict[str, object]:
    config = StreamingAttentionConfig(
        workspace_budget_bytes=workspace_mib * 2**20,
        kv_chunk_tokens=kv_chunk,
        backend=backend,
        num_kv_buffers=2,
        num_output_buffers=1,
    )
    plan = build_plan(
        q_heads=q.shape[1],
        kv_heads=k.shape[1],
        head_dim=q.shape[2],
        dtype=q.dtype,
        device="cuda",
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        config=config,
    )
    runner = StreamingAttentionRunner(plan, config)
    stats = StreamingAttentionStats()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    start_event.record()
    runner(q, k, v, cu, cu, out=output, stats=stats)
    end_event.record()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "status": "success",
        "seconds": seconds,
        "cuda_seconds": start_event.elapsed_time(end_event) / 1000.0,
        "effective_tflops": flop / seconds / 1e12,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "plan": {
            "q_chunk_tokens": plan.q_chunk_tokens,
            "kv_chunk_tokens": plan.kv_chunk_tokens,
            "q_passes": (q.shape[0] + plan.q_chunk_tokens - 1) // plan.q_chunk_tokens,
            "estimated_workspace_bytes": plan.estimated_workspace_bytes,
            "block_m": plan.block_m,
            "block_n": plan.block_n,
            "num_warps": plan.num_warps,
            "num_stages": plan.num_stages,
        },
        "streaming_stats": stats.as_dict(),
        "output_signature": output_signature(output),
    }
    del runner, plan, config
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A30 400K attention backends")
    parser.add_argument("--tokens", type=int, default=409_600)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--kv-chunk", type=int, default=8192)
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "status": "runtime_error",
        "configuration": dict(vars(args)),
    }
    result["configuration"]["output"] = str(args.output)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        torch.set_num_threads(1)
        dtype = torch.bfloat16
        shape = (args.tokens, args.heads, args.head_dim)
        preparation_started = time.perf_counter()
        q, k, v = make_host_tensors_parallel(
            (shape, shape, shape),
            dtype,
            seed=args.seed,
            workers=args.cpu_workers,
            chunk_tokens=args.cpu_chunk_tokens,
        )
        output = torch.empty(shape, dtype=dtype, pin_memory=True)
        cu = make_bounds(args.tokens, 1)
        result["data_preparation_seconds"] = time.perf_counter() - preparation_started
        result["status"] = "running"
        atomic_json(args.output, result)

        flop = 4 * args.heads * args.head_dim * args.tokens * args.tokens
        scale = args.head_dim**-0.5
        result["resident_fa2"] = run_resident_fa2(q, k, v, scale=scale, flop=flop)
        atomic_json(args.output, result)
        result["flash2_split_2g"] = run_streaming(
            q,
            k,
            v,
            cu,
            output,
            backend="fa2",
            workspace_mib=args.workspace_mib,
            kv_chunk=args.kv_chunk,
            flop=flop,
        )
        atomic_json(args.output, result)
        result["triton_2g"] = run_streaming(
            q,
            k,
            v,
            cu,
            output,
            backend="triton",
            workspace_mib=args.workspace_mib,
            kv_chunk=args.kv_chunk,
            flop=flop,
        )
        result["status"] = "success"
    except Exception as error:  # noqa: BLE001 - benchmark must preserve partial results
        result["status"] = (
            "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
