from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from seqattn_core import (
    H3BlockOps,
    H3DiTStats,
    H3MaterializedProjection,
    H3MaterializedRunner,
    H3SequenceMeta,
    ProjectedAttentionRunner,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_plan,
)
from seqattn_core.benchmarking.common import atomic_json

from seqattn_multigpu import (
    DynamicScheduleConfig,
    MultiGpuDeviceSpec,
    MultiGpuH3DiTStats,
    MultiGpuH3MaterializedRunner,
    build_multi_gpu_plan,
)


def _device_weight(
    shape: tuple[int, int],
    *,
    device: torch.device,
    seed: int,
    fan_in: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    bound = 1.0 / math.sqrt(fan_in)
    return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
        -bound,
        bound,
        generator=generator,
    )


def _make_ops(
    *,
    device: torch.device,
    hidden_host: torch.Tensor,
    hidden_features: int,
    heads: int,
    head_dim: int,
    mlp_features: int,
    seed: int,
) -> tuple[H3MaterializedProjection, H3BlockOps]:
    inner = heads * head_dim
    qkv_weight = _device_weight(
        (3 * inner, hidden_features),
        device=device,
        seed=seed + 1,
        fan_in=hidden_features,
    )
    out_weight = _device_weight(
        (hidden_features, inner),
        device=device,
        seed=seed + 2,
        fan_in=inner,
    )
    fc1_weight = _device_weight(
        (2 * mlp_features, hidden_features),
        device=device,
        seed=seed + 3,
        fan_in=hidden_features,
    )
    fc2_weight = _device_weight(
        (hidden_features, mlp_features),
        device=device,
        seed=seed + 4,
        fan_in=mlp_features,
    )

    def project_qkv(tile: torch.Tensor, start: int, stop: int):
        del start, stop
        qkv = F.linear(tile, qkv_weight).view(-1, 3, heads, head_dim)
        return tuple(qkv[:, index].contiguous() for index in range(3))

    def attention_epilogue(
        attention: torch.Tensor,
        residual_host: torch.Tensor,
        start: int,
        stop: int,
    ):
        residual = residual_host[start:stop].to(device, non_blocking=True)
        return F.linear(attention.reshape(stop - start, inner), out_weight).add_(residual)

    def mlp(post_attention: torch.Tensor, start: int, stop: int):
        del start, stop
        gate, up = F.linear(post_attention, fc1_weight).chunk(2, dim=-1)
        update = F.linear(F.silu(gate).mul_(up), fc2_weight)
        return post_attention.add_(update)

    return H3MaterializedProjection(project_qkv), H3BlockOps(attention_epilogue, mlp)


def _sync_devices(devices: list[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _output_sample(hidden_host: torch.Tensor) -> dict[str, object]:
    tokens = hidden_host.shape[0]
    token_indices = sorted({0, tokens // 4, tokens // 2, (3 * tokens) // 4, tokens - 1})
    feature_indices = (0, 17, 127, 1023, hidden_host.shape[1] - 1)
    sample = torch.stack(
        [hidden_host[token, list(feature_indices)].float() for token in token_indices]
    )
    if not torch.isfinite(sample).all():
        raise FloatingPointError("sampled H3 output contains NaN or Inf")
    return {
        "token_indices": token_indices,
        "feature_indices": list(feature_indices),
        "values": sample.tolist(),
        "sum": float(sample.sum()),
        "l2": float(torch.linalg.vector_norm(sample)),
    }


def _total_flops(
    *,
    tokens: int,
    hidden_features: int,
    heads: int,
    head_dim: int,
    mlp_features: int,
) -> int:
    inner = heads * head_dim
    qkv = 2 * tokens * hidden_features * (3 * inner)
    attention = 4 * heads * head_dim * tokens * tokens
    out_projection = 2 * tokens * inner * hidden_features
    fc1 = 2 * tokens * hidden_features * (2 * mlp_features)
    fc2 = 2 * tokens * mlp_features * hidden_features
    return qkv + attention + out_projection + fc1 + fc2


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one complete H3 block on 1 or 2 GPUs")
    parser.add_argument("--mode", choices=("single", "multi"), required=True)
    parser.add_argument("--tokens", type=int, default=132288)
    parser.add_argument("--hidden-features", type=int, default=5376)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-features", type=int, default=14336)
    parser.add_argument("--projection-chunk", type=int, default=4096)
    parser.add_argument("--q-chunk", type=int, default=5760)
    parser.add_argument("--q-capacity", type=int, default=12288)
    parser.add_argument("--kv-chunk", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
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
        },
    }
    runner: H3MaterializedRunner | MultiGpuH3MaterializedRunner | None = None
    try:
        required_devices = 1 if args.mode == "single" else 2
        if torch.cuda.device_count() != required_devices:
            raise RuntimeError(
                f"{args.mode} mode requires exactly {required_devices} visible CUDA devices"
            )
        devices = [torch.device(f"cuda:{index}") for index in range(required_devices)]
        result["environment"] |= {
            "devices": [torch.cuda.get_device_name(device) for device in devices],
            "device_count": required_devices,
        }

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        original = torch.empty(
            (args.tokens, args.hidden_features),
            dtype=torch.bfloat16,
            pin_memory=True,
        ).normal_(generator=generator)
        hidden = torch.empty_like(original, pin_memory=True)
        hidden.copy_(original)
        cu_seqlens = torch.tensor([0, args.tokens], dtype=torch.int32)
        sequence_meta = H3SequenceMeta(cu_seqlens)
        blocks = {
            str(device): _make_ops(
                device=device,
                hidden_host=hidden,
                hidden_features=args.hidden_features,
                heads=args.heads,
                head_dim=args.head_dim,
                mlp_features=args.mlp_features,
                seed=args.seed + 100,
            )
            for device in devices
        }

        if args.mode == "single":
            attention_config = StreamingAttentionConfig(
                q_chunk_tokens=args.q_chunk,
                kv_chunk_tokens=args.kv_chunk,
                output_mode="device_consumer",
                backend="triton",
            )
            attention_plan = build_plan(
                q_heads=args.heads,
                kv_heads=args.heads,
                head_dim=args.head_dim,
                dtype=torch.bfloat16,
                device=devices[0],
                max_q_tokens=args.tokens,
                max_kv_tokens=args.tokens,
                config=attention_config,
            )
            projected = ProjectedAttentionRunner(
                attention_plan,
                attention_config,
                ProjectionPipelineConfig(projection_chunk_tokens=args.projection_chunk),
            )
            runner = H3MaterializedRunner(
                projected,
                hidden_features=args.hidden_features,
                mlp_chunk_tokens=attention_plan.q_chunk_tokens,
            )
            result["plan"] = {
                "q_chunk_tokens": attention_plan.q_chunk_tokens,
                "kv_chunk_tokens": attention_plan.kv_chunk_tokens,
                "projection_chunk_tokens": args.projection_chunk,
                "estimated_workspace_bytes": runner.plan.estimated_workspace_bytes,
            }

            def run_once():
                stats = H3DiTStats()
                runner.run_block_(
                    hidden,
                    sequence_meta,
                    blocks[str(devices[0])][0],
                    blocks[str(devices[0])][1],
                    softmax_scale=args.head_dim**-0.5,
                    stats=stats,
                )
                return stats

        else:
            workspace_config = StreamingAttentionConfig(
                q_chunk_tokens=args.q_capacity,
                kv_chunk_tokens=args.kv_chunk,
                output_mode="device_consumer",
                backend="triton",
            )
            device_specs = [
                MultiGpuDeviceSpec(
                    device=device,
                    config=workspace_config,
                    compute_tflops=180.0,
                    h2d_gbps=36.0,
                    d2h_gbps=36.0,
                    q_chunk_tokens=args.q_chunk,
                    q_capacity_tokens=args.q_capacity,
                )
                for device in devices
            ]
            dynamic_config = DynamicScheduleConfig(enable_task_trace=False)
            multi_plan = build_multi_gpu_plan(
                q_heads=args.heads,
                kv_heads=args.heads,
                head_dim=args.head_dim,
                dtype=torch.bfloat16,
                max_q_tokens=args.tokens,
                max_kv_tokens=args.tokens,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                devices=device_specs,
                schedule_mode="dynamic",
                dynamic_config=dynamic_config,
            )
            primary_schedule = multi_plan.schedules[0]
            projected = ProjectedAttentionRunner(
                primary_schedule.attention_plan,
                primary_schedule.config,
                ProjectionPipelineConfig(projection_chunk_tokens=args.projection_chunk),
            )
            runner = MultiGpuH3MaterializedRunner(
                projected,
                multi_plan,
                hidden_features=args.hidden_features,
                projection_chunk_tokens=args.projection_chunk,
                dynamic_config=dynamic_config,
            )
            result["plan"] = {
                "initial_q_chunk_tokens": args.q_chunk,
                "q_capacity_tokens": args.q_capacity,
                "kv_chunk_tokens": args.kv_chunk,
                "projection_chunk_tokens": args.projection_chunk,
                "per_device_estimated_workspace_bytes": runner.per_device_estimated_workspace_bytes,
            }

            def run_once():
                stats = MultiGpuH3DiTStats()
                runner.run_block_(
                    hidden,
                    sequence_meta,
                    {device: block[0] for device, block in blocks.items()},
                    {device: block[1] for device, block in blocks.items()},
                    softmax_scale=args.head_dim**-0.5,
                    stats=stats,
                )
                return stats

        for _ in range(args.warmup):
            hidden.copy_(original)
            run_once()
            _sync_devices(devices)

        durations = []
        latest_stats: H3DiTStats | MultiGpuH3DiTStats | None = None
        for _ in range(args.repeats):
            hidden.copy_(original)
            started = time.perf_counter()
            latest_stats = run_once()
            _sync_devices(devices)
            durations.append(time.perf_counter() - started)

        assert latest_stats is not None
        mean_seconds = statistics.fmean(durations)
        total_flops = _total_flops(
            tokens=args.tokens,
            hidden_features=args.hidden_features,
            heads=args.heads,
            head_dim=args.head_dim,
            mlp_features=args.mlp_features,
        )
        result.update(
            status="success",
            seconds=durations,
            mean_seconds=mean_seconds,
            median_seconds=statistics.median(durations),
            min_seconds=min(durations),
            max_seconds=max(durations),
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=total_flops / mean_seconds / 1e12,
            total_flops=total_flops,
            output_sample=_output_sample(hidden),
            stats=latest_stats.as_dict(),
            torch_peak_allocated_mib={
                str(device): torch.cuda.max_memory_allocated(device) / 2**20 for device in devices
            },
            torch_peak_reserved_mib={
                str(device): torch.cuda.max_memory_reserved(device) / 2**20 for device in devices
            },
        )
    except Exception as error:  # noqa: BLE001 - benchmark must preserve failures in JSON.
        result["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if isinstance(runner, MultiGpuH3MaterializedRunner):
            runner.close()

    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
