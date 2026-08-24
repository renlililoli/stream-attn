from __future__ import annotations

import argparse
import os
import platform
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from seqattn_core import (
    CallbackOutputSink,
    MemoryPageSource,
    PagedAttentionConfig,
    PagedAttentionRunner,
    PagedAttentionStats,
    SimulatedNvmeConfig,
    SimulatedNvmeDevice,
    SimulatedPageSink,
    SimulatedPageSource,
    StreamingAttentionConfig,
)
from seqattn_core.benchmarking.common import ProcessMemorySampler, atomic_json, make_bounds


def make_tensors_parallel(
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    *,
    seed: int,
    workers: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, ...]:
    tensors = tuple(torch.empty(shape, dtype=dtype, pin_memory=True) for shape in shapes)
    tasks = []
    for tensor_index, tensor in enumerate(tensors):
        for chunk_index, start in enumerate(range(0, tensor.shape[0], chunk_tokens)):
            tasks.append(
                (
                    tensor_index,
                    chunk_index,
                    start,
                    min(start + chunk_tokens, tensor.shape[0]),
                )
            )

    def fill(task: tuple[int, int, int, int]) -> None:
        tensor_index, chunk_index, start, stop = task
        generator = torch.Generator(device="cpu").manual_seed(
            seed + tensor_index * 1_000_003 + chunk_index
        )
        tensors[tensor_index][start:stop].normal_(generator=generator)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seqattn-data") as pool:
        list(pool.map(fill, tasks))
    return tensors


class OutputSignature:
    def __init__(self, total_tokens: int, width: int = 8) -> None:
        self.indices = sorted(
            {0, total_tokens // 4, total_tokens // 2, 3 * total_tokens // 4, total_tokens - 1}
        )
        self.width = width
        self.values: dict[int, list[float]] = {}
        self._lock = threading.Lock()

    def consume(self, page, data: torch.Tensor) -> None:
        page_values = {}
        for index in self.indices:
            if page.token_start <= index < page.token_stop:
                local_index = index - page.token_start
                page_values[index] = data[local_index, 0, : self.width].float().tolist()
        if page_values:
            with self._lock:
                self.values.update(page_values)

    def from_gpu(self, output: torch.Tensor) -> None:
        for index in self.indices:
            self.values[index] = output[0, index, 0, : self.width].float().cpu().tolist()

    def from_cpu(self, output: torch.Tensor) -> None:
        for index in self.indices:
            self.values[index] = output[index, 0, : self.width].float().tolist()

    def as_dict(self) -> dict[str, list[float]]:
        missing = set(self.indices) - self.values.keys()
        if missing:
            raise RuntimeError(f"output signature is missing token indices: {sorted(missing)}")
        return {str(index): self.values[index] for index in self.indices}


def activation_sizes(
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> dict[str, int]:
    element_size = torch.empty((), dtype=dtype).element_size()
    q_bytes = tokens * q_heads * head_dim * element_size
    kv_tensor_bytes = tokens * kv_heads * head_dim * element_size
    return {
        "q_bytes": q_bytes,
        "k_bytes": kv_tensor_bytes,
        "v_bytes": kv_tensor_bytes,
        "qkv_bytes": q_bytes + 2 * kv_tensor_bytes,
        "output_bytes": q_bytes,
        "gpu_resident_qkv_output_bytes": 2 * q_bytes + 2 * kv_tensor_bytes,
    }


def run_gpu_resident(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
    signature: OutputSignature,
) -> tuple[float, float, dict[str, object]]:
    try:
        from flash_attn import flash_attn_func
    except ImportError as error:
        raise RuntimeError("gpu-resident mode requires flash-attn") from error

    residency_started = time.perf_counter()
    q_gpu = q.to("cuda", non_blocking=True)
    k_gpu = k.to("cuda", non_blocking=True)
    v_gpu = v.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    residency_seconds = time.perf_counter() - residency_started

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = flash_attn_func(
        q_gpu.unsqueeze(0),
        k_gpu.unsqueeze(0),
        v_gpu.unsqueeze(0),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()
    execution_seconds = time.perf_counter() - started
    signature.from_gpu(output)
    return (
        residency_seconds,
        execution_seconds,
        {
            "output_residency": "gpu",
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    )


def run_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    *,
    mode: str,
    args: argparse.Namespace,
    storage_dtype: str,
    signature: OutputSignature,
) -> tuple[float, float, dict[str, object]]:
    q_source = MemoryPageSource(
        q=q,
        cu_seqlens_q=cu,
        page_target_bytes=int(args.q_page_mib * 2**20),
        block_n=64,
    )
    kv_source = MemoryPageSource(
        k=k,
        v=v,
        cu_seqlens_k=cu,
        page_target_bytes=int(args.kv_page_mib * 2**20),
        block_n=64,
        kv_storage_dtype=storage_dtype,
    )
    sink = CallbackOutputSink(signature.consume)
    simulated_config = None
    if mode == "simulated-nvme":
        simulated_config = SimulatedNvmeConfig(
            read_bandwidth_bytes_per_second=args.simulate_read_gbps * 1e9,
            write_bandwidth_bytes_per_second=args.simulate_write_gbps * 1e9,
            read_latency_seconds=args.simulate_read_latency_us * 1e-6,
            write_latency_seconds=args.simulate_write_latency_us * 1e-6,
            max_concurrent_reads=args.simulate_max_concurrent_reads,
            max_concurrent_writes=args.simulate_max_concurrent_writes,
            jitter_fraction=0.0,
            random_seed=args.seed,
        )
        device = SimulatedNvmeDevice(simulated_config)
        q_source = SimulatedPageSource(q_source, device=device)
        kv_source = SimulatedPageSource(kv_source, device=device)
        sink = SimulatedPageSink(sink, device=device)

    attention_config = StreamingAttentionConfig(
        workspace_budget_bytes=int(args.workspace_gib * 2**30),
        q_chunk_tokens=args.q_chunk,
        kv_chunk_tokens=args.kv_chunk,
        backend="triton",
        block_m=args.block_m,
        block_n=args.block_n,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        num_kv_buffers=args.num_kv_buffers,
        num_output_buffers=args.num_output_buffers,
    )
    config = PagedAttentionConfig(
        attention=attention_config,
        host_memory_budget_bytes=int(args.host_budget_gib * 2**30),
        pinned_staging_budget_bytes=1 * 2**30,
        direct_io_bounce_budget_bytes=512 * 2**20,
        metadata_margin_bytes=128 * 2**20,
        page_target_bytes=int(args.kv_page_mib * 2**20),
        io_workers=args.io_workers,
        io_queue_depth=args.queue_depth,
        num_output_buffers=args.num_output_buffers,
        direct_io=False,
        kv_storage_dtype=storage_dtype,
    )
    runner = PagedAttentionRunner(config, device="cuda")
    stats = PagedAttentionStats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    pages_written = runner.run(
        q_source,
        kv_source,
        cu,
        cu,
        sink,
        causal=args.causal,
        stats=stats,
    )
    torch.cuda.synchronize()
    execution_seconds = time.perf_counter() - started
    details: dict[str, object] = {
        "output_residency": "callback",
        "output_pages_written": pages_written,
        "q_page_count": len(q_source.q_pages),
        "kv_page_count": len(kv_source.kv_pages),
        "paged_stats": stats.as_dict(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if simulated_config is not None:
        details["simulated_nvme_config"] = simulated_config.as_dict()
    return 0.0, execution_seconds, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare >8GiB attention storage tiers")
    parser.add_argument("--mode", choices=("gpu-resident", "dram", "simulated-nvme"), required=True)
    parser.add_argument("--tokens", type=int, default=524_288)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--host-budget-gib", type=float, default=8.0)
    parser.add_argument("--q-page-mib", type=float, default=448.0)
    parser.add_argument("--kv-page-mib", type=float, default=16.0)
    parser.add_argument("--q-chunk", type=int)
    parser.add_argument("--kv-chunk", type=int, default=8192)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8))
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--num-kv-buffers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--num-output-buffers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--simulate-read-gbps", type=float, default=7.0)
    parser.add_argument("--simulate-write-gbps", type=float, default=6.0)
    parser.add_argument("--simulate-read-latency-us", type=float, default=80.0)
    parser.add_argument("--simulate-write-latency-us", type=float, default=100.0)
    parser.add_argument("--simulate-max-concurrent-reads", type=int, default=4)
    parser.add_argument("--simulate-max-concurrent-writes", type=int, default=4)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
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
        "storage_performance_valid": False,
        "storage_performance_note": (
            "The NVMe tier is an in-memory timing simulation, not a physical-device result."
        ),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        if args.cpu_workers <= 0 or args.cpu_chunk_tokens <= 0:
            raise ValueError("CPU generation worker and chunk counts must be positive")
        torch.set_num_threads(1)
        dtype = getattr(torch, args.dtype)
        sizes = activation_sizes(args.tokens, args.q_heads, args.kv_heads, args.head_dim, dtype)
        result["activation_sizes"] = sizes
        if sizes["qkv_bytes"] <= 8 * 2**30:
            raise ValueError("the selected complete Q/K/V activation is not larger than 8GiB")

        torch.cuda.reset_peak_memory_stats()
        with ProcessMemorySampler(args.sample_interval_ms / 1000) as sampler:
            preparation_started = time.perf_counter()
            q, k, v = make_tensors_parallel(
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
            cu = make_bounds(args.tokens, args.segments)
            data_preparation_seconds = time.perf_counter() - preparation_started
            signature = OutputSignature(args.tokens)
            storage_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
            if args.mode == "gpu-resident":
                residency_seconds, execution_seconds, details = run_gpu_resident(
                    q,
                    k,
                    v,
                    causal=args.causal,
                    softmax_scale=args.head_dim**-0.5,
                    signature=signature,
                )
            else:
                residency_seconds, execution_seconds, details = run_paged(
                    q,
                    k,
                    v,
                    cu,
                    mode=args.mode,
                    args=args,
                    storage_dtype=storage_dtype,
                    signature=signature,
                )
        lengths = torch.diff(cu).tolist()
        flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in lengths)
        result.update(
            status="success",
            data_preparation_seconds=data_preparation_seconds,
            data_preparation_gib_per_second=(sizes["qkv_bytes"] / 2**30 / data_preparation_seconds),
            gpu_residency_preparation_seconds=residency_seconds,
            execution_seconds=execution_seconds,
            effective_tflops=flop / execution_seconds / 1e12,
            tokens_per_second=args.tokens / execution_seconds,
            process_peak_rss_bytes=sampler.peak_rss_bytes,
            nvml_process_peak_bytes=sampler.peak_vram_bytes,
            memory_samples=sampler.samples,
            output_signature=signature.as_dict(),
            **details,
        )
    except Exception as error:  # noqa: BLE001 - benchmark must publish failures as JSON
        result["status"] = (
            "oom" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    atomic_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
