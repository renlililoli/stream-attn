from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from seqattn_core import StreamingAttentionConfig
from seqattn_core.benchmarking.common import (
    atomic_json,
    make_host_tensors_parallel,
    make_pinned_host_tensors_parallel,
)

from seqattn_multigpu import (
    DynamicScheduleConfig,
    MultiGpuAttentionStats,
    MultiGpuDeviceSpec,
    MultiGpuStreamingAttentionRunner,
    build_multi_gpu_plan,
)


def _csv_numbers(value: str, count: int, cast):
    values = [cast(item) for item in value.split(",")]
    if len(values) != count:
        raise ValueError(f"expected {count} comma-separated values, got {value!r}")
    return values


def _command_output(command: list[str]) -> str:
    return subprocess.run(command, check=False, capture_output=True, text=True).stdout.strip()


def _output_signature(output: torch.Tensor) -> dict[str, list[float]]:
    indices = (0, output.shape[0] // 4, output.shape[0] // 2, 3 * output.shape[0] // 4, -1)
    return {str(index): output[index, 0, :8].float().tolist() for index in indices}


def _synchronize(devices: list[str]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 524K static or dynamic multi-GPU Q")
    parser.add_argument("--mode", choices=("static", "dynamic"), required=True)
    parser.add_argument("--devices", required=True, help="CUDA devices, for example cuda:0,cuda:1")
    parser.add_argument("--q-chunks", required=True)
    parser.add_argument("--q-min", default="2048")
    parser.add_argument("--q-capacities")
    parser.add_argument("--compute-tflops", required=True)
    parser.add_argument("--h2d-gbps", required=True)
    parser.add_argument("--d2h-gbps")
    parser.add_argument("--tokens", type=int, default=524288)
    parser.add_argument("--kv-chunk", type=int, default=8192)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--cpu-workers", type=int, default=32)
    parser.add_argument("--cpu-chunk-tokens", type=int, default=4096)
    parser.add_argument("--dynamic-capacity-default", type=int, default=23040)
    parser.add_argument("--task-trace", action="store_true")
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
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                    "--format=csv,noheader",
                ]
            ).splitlines(),
            "numa_policy": _command_output(["numactl", "--show"]).splitlines(),
        },
    }
    runner = None
    try:
        devices = args.devices.split(",")
        count = len(devices)
        q_chunks = _csv_numbers(args.q_chunks, count, int)
        q_min = (
            _csv_numbers(args.q_min, count, int) if "," in args.q_min else [int(args.q_min)] * count
        )
        capacities = (
            _csv_numbers(args.q_capacities, count, int)
            if args.q_capacities
            else ([args.dynamic_capacity_default] * count if args.mode == "dynamic" else q_chunks)
        )
        compute_tflops = _csv_numbers(args.compute_tflops, count, float)
        h2d_gbps = _csv_numbers(args.h2d_gbps, count, float)
        d2h_gbps = _csv_numbers(args.d2h_gbps, count, float) if args.d2h_gbps else h2d_gbps
        if args.calls <= 0 or args.warmup < 0:
            raise ValueError("calls must be positive and warmup must be non-negative")

        dtype = torch.bfloat16
        preparation_started = time.perf_counter()
        shape = (args.tokens, 56, 128)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="seqattn-prepare") as pool:
            inputs = pool.submit(
                make_host_tensors_parallel,
                (shape, shape, shape),
                dtype,
                seed=0,
                workers=args.cpu_workers,
                chunk_tokens=args.cpu_chunk_tokens,
            )
            output_future = pool.submit(
                make_pinned_host_tensors_parallel,
                (shape,),
                dtype,
                workers=1,
            )
            q, k, v = inputs.result()
            (output_buffer,) = output_future.result()
        result["data_preparation_seconds"] = time.perf_counter() - preparation_started

        cu = torch.tensor([0, args.tokens], dtype=torch.int32)
        specs = []
        for index, device in enumerate(devices):
            specs.append(
                MultiGpuDeviceSpec(
                    device=device,
                    config=StreamingAttentionConfig(
                        q_chunk_tokens=q_chunks[index],
                        kv_chunk_tokens=args.kv_chunk,
                        block_m=128,
                        block_n=64,
                        num_warps=8,
                        num_stages=3,
                        num_kv_buffers=2,
                        num_output_buffers=1,
                        backend="triton",
                    ),
                    compute_tflops=compute_tflops[index],
                    h2d_gbps=h2d_gbps[index],
                    d2h_gbps=d2h_gbps[index],
                    q_min_tokens=q_min[index],
                    q_capacity_tokens=capacities[index],
                )
            )
        dynamic_config = DynamicScheduleConfig(enable_task_trace=args.task_trace)
        plan = build_multi_gpu_plan(
            q_heads=56,
            kv_heads=56,
            head_dim=128,
            dtype=dtype,
            max_q_tokens=args.tokens,
            max_kv_tokens=args.tokens,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            devices=specs,
            schedule_mode=args.mode,
            dynamic_config=dynamic_config,
        )
        result["plan"] = {
            "schedule_mode": plan.schedule_mode,
            "estimated_makespan_seconds": plan.estimated_makespan_seconds,
            "schedules": [
                {
                    "device": str(schedule.device),
                    "q_range": [schedule.q_range_start, schedule.q_range_stop],
                    "q_tokens": schedule.q_tokens,
                    "q_chunk_tokens": schedule.initial_q_tokens,
                    "q_min_tokens": schedule.q_min_tokens,
                    "q_capacity_tokens": schedule.q_capacity_tokens,
                    "kv_chunk_tokens": schedule.attention_plan.kv_chunk_tokens,
                    "estimated_seconds": schedule.estimated_seconds,
                    "task_count": len(schedule.query_tasks),
                    "workspace_bytes": schedule.attention_plan.estimated_workspace_bytes,
                }
                for schedule in plan.schedules
            ],
        }
        runner = MultiGpuStreamingAttentionRunner(plan)

        for _ in range(args.warmup):
            runner(q, k, v, cu, cu, out=output_buffer)
            _synchronize(devices)

        calls = []
        flop = 4 * 56 * 128 * args.tokens * args.tokens
        for call_index in range(args.calls):
            stats = MultiGpuAttentionStats()
            started = time.perf_counter()
            runner(q, k, v, cu, cu, out=output_buffer, stats=stats)
            _synchronize(devices)
            elapsed = time.perf_counter() - started
            calls.append(
                {
                    "call": call_index + 1,
                    "seconds": elapsed,
                    "effective_tflops": flop / elapsed / 1e12,
                    "stats": stats.as_dict(),
                }
            )
            print(json.dumps(calls[-1], sort_keys=True), flush=True)

        seconds = [item["seconds"] for item in calls]
        assert all(isinstance(item, float) for item in seconds)
        result.update(
            status="success",
            calls=calls,
            median_seconds=statistics.median(seconds),
            median_effective_tflops=flop / statistics.median(seconds) / 1e12,
            tokens_per_second=args.tokens / statistics.median(seconds),
            output_signature=_output_signature(output_buffer),
            all_finite=bool(torch.isfinite(output_buffer).all()),
            torch_peak_allocated_bytes={
                device: torch.cuda.max_memory_allocated(device) for device in devices
            },
            torch_peak_reserved_bytes={
                device: torch.cuda.max_memory_reserved(device) for device in devices
            },
        )
    except Exception as error:  # noqa: BLE001 - benchmark preserves failures in JSON.
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if runner is not None:
            runner.close()
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
