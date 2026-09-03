from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from seqattn_core import StreamingAttentionConfig, StreamingAttentionRunner, build_attention_plan
from seqattn_core.config import ProjectionPipelineConfig
from seqattn_core.projection import ProjectedAttentionRunner
from seqattn_core.sparse import (
    SolStreamingAttentionRunner,
    SolStreamingStats,
    build_sol_streaming_plan,
)
from seqattn_core.sparse.materialized import SolMaterializedSource


class Collector:
    def __init__(self, tokens: int, heads: int, device: torch.device) -> None:
        self.output = torch.empty(
            (tokens, heads * 128),
            dtype=torch.bfloat16,
            device=device,
        )

    def __call__(self, tile: torch.Tensor, start: int, stop: int) -> None:
        self.output[start:stop].copy_(tile)

    def finish(self) -> None:
        pass

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.output.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact, official Sol, BF16 streamed Sol, and INT8 streamed Sol outputs"
    )
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prefix", type=int, default=1495)
    parser.add_argument("--q-chunk", type=int, default=4096)
    parser.add_argument("--kv-chunk", type=int, default=4096)
    parser.add_argument("--projection-tile", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_host_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.tokens, args.heads, 128)
    return tuple(
        torch.randn(shape, generator=generator, dtype=torch.bfloat16).pin_memory() for _ in range(3)
    )


def make_runners(args: argparse.Namespace):
    plan = build_attention_plan(
        q_heads=args.heads,
        kv_heads=args.heads,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda:0",
        max_q_tokens=args.tokens,
        max_kv_tokens=args.tokens,
        config=StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=args.q_chunk,
            kv_chunk_tokens=args.kv_chunk,
            output_mode="device_consumer",
        ),
    )
    sol_plan = build_sol_streaming_plan(plan)
    dense = StreamingAttentionRunner(sol_plan.attention)
    return sol_plan, dense, SolStreamingAttentionRunner(sol_plan, dense)


def run_current_bf16(
    args: argparse.Namespace,
    host: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, SolStreamingStats, int]:
    sol_plan, _, runner = make_runners(args)
    collector = Collector(args.tokens, args.heads, sol_plan.attention.device)
    stats = SolStreamingStats()
    runner.run_with_device_consumer(
        *host,
        torch.tensor([0, args.tokens], dtype=torch.int32),
        exact_prefix_tokens=(args.prefix,),
        output_consumer=collector,
        tau=args.tau,
        stats=stats,
    )
    return (
        collector.output.view(args.tokens, args.heads, 128),
        stats,
        sol_plan.estimated_workspace_bytes,
    )


def run_current_int8(
    args: argparse.Namespace,
    host: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, SolStreamingStats, int]:
    sol_plan = build_sol_streaming_plan(
        build_attention_plan(
            q_heads=args.heads,
            kv_heads=args.heads,
            head_dim=128,
            dtype=torch.bfloat16,
            device="cuda:0",
            max_q_tokens=args.tokens,
            max_kv_tokens=args.tokens,
            config=StreamingAttentionConfig(
                backend="triton",
                q_chunk_tokens=args.q_chunk,
                kv_chunk_tokens=args.kv_chunk,
                output_mode="device_consumer",
            ),
        )
    )
    projected = ProjectedAttentionRunner(
        sol_plan.attention,
        ProjectionPipelineConfig(projection_tile_tokens=args.projection_tile),
    )
    runner = SolStreamingAttentionRunner(sol_plan, projected.attention)
    source = SolMaterializedSource(
        sol_plan,
        runner.workspace.dense,
        k_storage=projected.arena.k,
        v_storage=projected.arena.v,
        pin_memory=projected.pipeline_config.pin_qkv,
    )
    bounds = [0, args.tokens]
    ranges = source.prepare(
        projected.arena.q,
        bounds,
        projection_tile_tokens=args.projection_tile,
    )
    device = sol_plan.attention.device
    for start, stop in ranges:
        q, k, v = (tensor[start:stop].to(device, non_blocking=True) for tensor in host)
        source.copy_to_host(start, stop, source.encode(q, k, v, start, stop))
    torch.cuda.synchronize(device)

    collector = Collector(args.tokens, args.heads, device)
    stats = SolStreamingStats()
    runner.run_with_qkv_source(
        source,
        args.tokens,
        torch.tensor([0, args.tokens], dtype=torch.int32),
        exact_prefix_tokens=(args.prefix,),
        output_consumer=collector,
        tau=args.tau,
        stats=stats,
    )
    return (
        collector.output.view(args.tokens, args.heads, 128),
        stats,
        sol_plan.estimated_workspace_bytes,
    )


def run_official(
    args: argparse.Namespace,
    host: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, str]:
    from sol_attn import get_sol_attn_backend, sol_attn

    q, k, v = (tensor.to("cuda:0", non_blocking=True).unsqueeze(0) for tensor in host)
    torch.cuda.synchronize()
    output = sol_attn(
        q,
        k,
        v,
        tau=args.tau,
        thresh_type="diag",
        sink_start=0 if args.prefix else None,
        sink_tokens=args.prefix,
    )
    exact_q_tokens = min(args.tokens, math.ceil(args.prefix / 64) * 64)
    if exact_q_tokens:
        dense_prefix = torch.nn.functional.scaled_dot_product_attention(
            q[:, :exact_q_tokens].transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=128**-0.5,
        ).transpose(1, 2)
        output[:, :exact_q_tokens].copy_(dense_prefix)
    torch.cuda.synchronize()
    del q, k, v
    torch.cuda.empty_cache()
    return output.squeeze(0), get_sol_attn_backend("cuda:0")


def run_dense(
    host: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    q, k, v = (tensor.to("cuda:0", non_blocking=True).unsqueeze(0) for tensor in host)
    torch.cuda.synchronize()
    output = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=128**-0.5,
    ).transpose(1, 2)
    torch.cuda.synchronize()
    del q, k, v
    torch.cuda.empty_cache()
    return output.squeeze(0)


def error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    start: int,
    stop: int,
    chunk_tokens: int = 512,
) -> dict[str, float | int | None]:
    count = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_diff_sq = 0.0
    sum_actual_sq = 0.0
    sum_reference_sq = 0.0
    dot = 0.0
    for chunk_start in range(start, stop, chunk_tokens):
        chunk_stop = min(chunk_start + chunk_tokens, stop)
        got = actual[chunk_start:chunk_stop].float()
        expected = reference[chunk_start:chunk_stop].float()
        difference = got - expected
        count += difference.numel()
        max_abs = max(max_abs, float(difference.abs().max()))
        sum_abs += float(difference.abs().sum())
        sum_diff_sq += float(torch.sum(difference * difference))
        sum_actual_sq += float(torch.sum(got * got))
        sum_reference_sq += float(torch.sum(expected * expected))
        dot += float(torch.sum(got * expected))
    if count == 0:
        return {"elements": 0}
    denominator = math.sqrt(sum_actual_sq * sum_reference_sq)
    return {
        "elements": count,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
        "rmse": math.sqrt(sum_diff_sq / count),
        "reference_rms": math.sqrt(sum_reference_sq / count),
        "relative_l2": (math.sqrt(sum_diff_sq / sum_reference_sq) if sum_reference_sq else None),
        "cosine_similarity": dot / denominator if denominator else None,
    }


def compare_regions(
    actual: torch.Tensor,
    reference: torch.Tensor,
    exact_q_tokens: int,
) -> dict[str, dict[str, float | int | None]]:
    tokens = actual.shape[0]
    return {
        "all_tokens": error_metrics(actual, reference, start=0, stop=tokens),
        "exact_query_prefix": error_metrics(
            actual,
            reference,
            start=0,
            stop=exact_q_tokens,
        ),
        "sparse_query_tail": error_metrics(
            actual,
            reference,
            start=exact_q_tokens,
            stop=tokens,
        ),
    }


def stats_dict(stats: SolStreamingStats, workspace_bytes: int) -> dict[str, object]:
    return {
        "backend": stats.backend,
        "q_chunks": stats.q_chunks,
        "kv_tiles": stats.kv_tiles,
        "h2d_bytes": stats.h2d_bytes,
        "exact_route_blocks": stats.exact_route_blocks,
        "approximate_route_blocks": stats.approximate_route_blocks,
        "effective_density": stats.effective_density,
        "estimated_workspace_bytes": workspace_bytes,
    }


def git_revision() -> str:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        cwd=repository,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    return f"unknown ({completed.stderr.strip()})"


def validate_runtime(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Sol accuracy comparison requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one idle GPU so cuda:0 maps to --physical-gpu")
    if args.physical_gpu < 0:
        raise ValueError("physical-gpu must be non-negative")
    if args.tokens <= 0 or args.heads <= 0:
        raise ValueError("tokens and heads must be positive")
    if args.q_chunk <= 0 or args.kv_chunk <= 0 or args.projection_tile <= 0:
        raise ValueError("chunk and projection tile sizes must be positive")
    if not math.isfinite(args.tau):
        raise ValueError("tau must be finite")
    if not 0 <= args.prefix <= args.tokens:
        raise ValueError("prefix must fit the sequence")


def main() -> None:
    args = parse_args()
    validate_runtime(args)
    exact_q_tokens = min(args.tokens, math.ceil(args.prefix / 64) * 64)
    report: dict[str, object] = {
        "status": "running",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "tokens": args.tokens,
        "heads": args.heads,
        "head_dim": 128,
        "dtype": "bfloat16",
        "tau": args.tau,
        "exact_prefix_tokens": args.prefix,
        "rounded_exact_query_tokens": exact_q_tokens,
        "q_chunk_tokens": args.q_chunk,
        "kv_chunk_tokens": args.kv_chunk,
        "projection_tile_tokens": args.projection_tile,
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "seqattn_revision": git_revision(),
        "official_revision": "d0c0a4685ab5dc2336d18b7213d85f13def92418",
        "device": torch.cuda.get_device_name(0),
    }
    try:
        host = make_host_inputs(args)
        outputs: dict[str, torch.Tensor] = {}
        if not args.skip_official:
            outputs["official"], report["official_backend"] = run_official(args, host)
        if args.include_dense:
            outputs["dense"] = run_dense(host)
        outputs["current_bf16"], bf16_stats, bf16_workspace = run_current_bf16(args, host)
        outputs["current_int8"], int8_stats, int8_workspace = run_current_int8(args, host)

        comparisons = {
            "int8_vs_bf16": compare_regions(
                outputs["current_int8"], outputs["current_bf16"], exact_q_tokens
            )
        }
        if "official" in outputs:
            comparisons["bf16_vs_official"] = compare_regions(
                outputs["current_bf16"], outputs["official"], exact_q_tokens
            )
            comparisons["int8_vs_official"] = compare_regions(
                outputs["current_int8"], outputs["official"], exact_q_tokens
            )
        if "dense" in outputs:
            for name in ("official", "current_bf16", "current_int8"):
                if name in outputs:
                    comparisons[f"{name}_vs_dense"] = compare_regions(
                        outputs[name], outputs["dense"], exact_q_tokens
                    )
        report["comparisons"] = comparisons
        report["current_bf16_stats"] = stats_dict(bf16_stats, bf16_workspace)
        report["current_int8_stats"] = stats_dict(int8_stats, int8_workspace)
        report["status"] = "ok"
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
