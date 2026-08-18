from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

import torch

from .benchmark import MemorySampler, atomic_json, configure_allocator, make_bounds
from .config import ProjectionPipelineConfig, StreamingAttentionConfig
from .pipeline import ProjectedAttentionRunner
from .planner import build_plan
from .stats import ProjectedAttentionStats, StreamingAttentionStats


def make_host_tensor(shape, dtype, generator):
    tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
    tensor.normal_(generator=generator)
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hidden -> QKV -> streaming attention -> output projection"
    )
    parser.add_argument("--mode", choices=("pipeline", "staged"), default="pipeline")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=5376)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--projection-chunk", type=int, default=2048)
    parser.add_argument("--q-chunk", type=int)
    parser.add_argument("--kv-chunk", type=int, default=4096)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--target-vram-mib", type=int)
    parser.add_argument("--safety-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=10)
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
        result["memory_policy"] = configure_allocator(
            args.target_vram_mib, args.safety_mib
        )
        dtype = getattr(torch, args.dtype)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        hidden_cpu = make_host_tensor(
            (args.tokens, args.hidden_size), dtype, generator
        )
        cu = make_bounds(args.tokens, args.segments)
        inner = args.heads * args.head_dim

        torch.manual_seed(args.seed + 1)
        qkv_linear = torch.nn.Linear(
            args.hidden_size, inner * 3, bias=False, device="cuda", dtype=dtype
        )
        out_linear = torch.nn.Linear(
            inner, args.hidden_size, bias=False, device="cuda", dtype=dtype
        )

        def project_qkv(hidden, start, stop):
            del start, stop
            qkv = qkv_linear(hidden).view(-1, args.heads, 3, args.head_dim)
            return (
                qkv[:, :, 0, :].contiguous(),
                qkv[:, :, 1, :].contiguous(),
                qkv[:, :, 2, :].contiguous(),
            )

        def output_projector(attention, start, stop):
            del start, stop
            return out_linear(attention)

        attention_config = StreamingAttentionConfig(
            workspace_budget_bytes=args.workspace_mib * 2**20,
            q_chunk_tokens=args.q_chunk,
            kv_chunk_tokens=args.kv_chunk,
            num_output_buffers=2,
            backend="triton",
            enable_nvtx=args.nvtx,
        )
        pipeline_config = ProjectionPipelineConfig(
            projection_chunk_tokens=args.projection_chunk,
            num_projection_buffers=2,
            enable_nvtx=args.nvtx,
        )
        plan = build_plan(
            q_heads=args.heads,
            kv_heads=args.heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device="cuda",
            max_q_tokens=args.tokens,
            max_kv_tokens=args.tokens,
            config=attention_config,
        )
        runner = ProjectedAttentionRunner(plan, attention_config, pipeline_config)
        output_cpu = torch.empty(
            (args.tokens, args.hidden_size), dtype=dtype, pin_memory=True
        )
        raw_attention_cpu = None
        if args.mode == "staged":
            raw_attention_cpu = torch.empty(
                (args.tokens, args.heads, args.head_dim),
                dtype=dtype,
                pin_memory=True,
            )

        result["plan"] = {
            "q_chunk_tokens": plan.q_chunk_tokens,
            "kv_chunk_tokens": plan.kv_chunk_tokens,
            "projection_chunk_tokens": args.projection_chunk,
            "estimated_attention_workspace_mib": plan.estimated_workspace_bytes / 2**20,
        }
        latest_stats: dict[str, object] | None = None

        def run_once():
            nonlocal latest_stats
            if args.mode == "pipeline":
                stats = ProjectedAttentionStats()
                output = runner(
                    hidden_cpu,
                    cu,
                    project_qkv=project_qkv,
                    output_projector=output_projector,
                    out=output_cpu,
                    causal=args.causal,
                    stats=stats,
                )
                latest_stats = stats.as_dict()
                return output

            projection_stats = ProjectedAttentionStats()
            q_cpu, k_cpu, v_cpu = runner.project_qkv_to_host(
                hidden_cpu, project_qkv, projection_stats
            )
            attention_stats = StreamingAttentionStats()
            runner.attention(
                q_cpu,
                k_cpu,
                v_cpu,
                cu,
                cu,
                causal=args.causal,
                out=raw_attention_cpu,
                stats=attention_stats,
            )
            output_h2d_bytes = 0
            output_d2h_bytes = 0
            for start in range(0, args.tokens, plan.q_chunk_tokens):
                stop = min(start + plan.q_chunk_tokens, args.tokens)
                attention_gpu = raw_attention_cpu[start:stop].to(
                    "cuda", non_blocking=True
                )
                projected = out_linear(attention_gpu.reshape(stop - start, inner))
                output_cpu[start:stop].copy_(projected, non_blocking=True)
                output_h2d_bytes += attention_gpu.numel() * attention_gpu.element_size()
                output_d2h_bytes += projected.numel() * projected.element_size()
            torch.cuda.synchronize()
            latest_stats = {
                "projection": projection_stats.as_dict(),
                "attention": attention_stats.as_dict(),
                "raw_attention_output_h2d_bytes": output_h2d_bytes,
                "projected_output_d2h_bytes": output_d2h_bytes,
                "total_h2d_bytes": (
                    projection_stats.projection_hidden_h2d_bytes
                    + attention_stats.h2d_bytes
                    + output_h2d_bytes
                ),
                "total_d2h_bytes": (
                    projection_stats.projection_qkv_d2h_bytes
                    + attention_stats.d2h_bytes
                    + output_d2h_bytes
                ),
            }
            return output_cpu

        for _ in range(args.warmup):
            run_once()
        torch.cuda.synchronize()
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = run_once()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with MemorySampler(args.sample_interval_ms / 1000) as sampler:
            probe_started = time.perf_counter()
            output = run_once()
            torch.cuda.synchronize()
            memory_probe_seconds = time.perf_counter() - probe_started
        if not torch.isfinite(output).all():
            raise FloatingPointError("pipeline output contains NaN or Inf")

        lengths = torch.diff(cu).tolist()
        qkv_flops = 2 * args.tokens * args.hidden_size * (3 * inner)
        attention_flops = 4 * args.heads * args.head_dim * sum(
            length * length for length in lengths
        )
        out_flops = 2 * args.tokens * inner * args.hidden_size
        total_flops = qkv_flops + attention_flops + out_flops
        mean_seconds = sum(durations) / len(durations)
        result.update(
            status="success",
            seconds=durations,
            mean_seconds=mean_seconds,
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=total_flops / mean_seconds / 1e12,
            torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            nvml_process_peak_mib=sampler.peak_mib,
            nvml_sample_count=sampler.samples,
            memory_probe_seconds=memory_probe_seconds,
            pipeline_stats=latest_stats,
        )
    except BaseException as error:
        result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
