from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import traceback
from pathlib import Path

import torch

from ..config import PagedAttentionConfig, StreamingAttentionConfig
from ..paged import (
    CallbackOutputSink,
    KVLayout,
    MemoryPageSource,
    PagedAttentionRunner,
    SimulatedNvmeConfig,
    SimulatedNvmeDevice,
    SimulatedPageSink,
    SimulatedPageSource,
    TensorLayout,
)
from ..stats import PagedAttentionStats
from ..storage import NvmeOutputSink, NvmeQKVWriter
from .common import ProcessMemorySampler, atomic_json, make_bounds, make_host_tensor


def make_tensor(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator):
    return make_host_tensor(shape, dtype, generator)


def stream_store(
    path: Path,
    *,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor,
    page_target_bytes: int,
    kv_storage_dtype: str,
    direct_io: bool,
    seed: int,
):
    layout_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    writer = NvmeQKVWriter(
        path,
        q_layout=TensorLayout(tokens, q_heads, head_dim, layout_name),
        kv_layout=KVLayout(
            tokens,
            kv_heads,
            head_dim,
            layout_name,
            kv_storage_dtype,
        ),
        cu_seqlens_q=cu_seqlens.tolist(),
        cu_seqlens_k=cu_seqlens.tolist(),
        page_target_bytes=page_target_bytes,
        block_n=64,
        direct_io=direct_io,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def q_pages():
        for page in writer.q_pages:
            yield torch.empty((page.valid_tokens, q_heads, head_dim), dtype=dtype).normal_(
                generator=generator
            )

    def kv_pages():
        for page in writer.kv_pages:
            shape = (page.valid_tokens, kv_heads, head_dim)
            yield (
                torch.empty(shape, dtype=dtype).normal_(generator=generator),
                torch.empty(shape, dtype=dtype).normal_(generator=generator),
            )

    started = time.perf_counter()
    store = writer.write_pages(q_pages(), kv_pages())
    return store, time.perf_counter() - started, writer.quantization_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark paged seqattn")
    parser.add_argument(
        "--storage",
        choices=("memory", "simulated-nvme", "nvme-bf16", "nvme-int8"),
        required=True,
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--host-budget-gib", type=float, default=8.0)
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    parser.add_argument("--page-mib", type=int, default=16)
    parser.add_argument("--q-page-mib", type=int)
    parser.add_argument("--kv-page-mib", type=int)
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--q-chunk", type=int)
    parser.add_argument("--kv-chunk", type=int)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--block-n", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--num-warps", type=int, choices=(2, 4, 8))
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--num-kv-buffers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--num-output-buffers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--buffered-io-for-tests", action="store_true")
    parser.add_argument("--formal-local-nvme", action="store_true")
    parser.add_argument("--simulate-read-gbps", type=float, default=7.0)
    parser.add_argument("--simulate-write-gbps", type=float, default=6.0)
    parser.add_argument("--simulate-read-latency-us", type=float, default=80.0)
    parser.add_argument("--simulate-write-latency-us", type=float, default=100.0)
    parser.add_argument("--simulate-max-concurrent-reads", type=int, default=4)
    parser.add_argument("--simulate-max-concurrent-writes", type=int, default=4)
    parser.add_argument("--simulate-jitter-fraction", type=float, default=0.0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--cuda-profiler-repeat", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    formal_nvme = args.storage.startswith("nvme-") and args.formal_local_nvme
    if args.storage == "simulated-nvme":
        performance_note = (
            "In-memory timing simulation only; it is not a filesystem, firmware, "
            "PCIe, or physical NVMe performance result."
        )
    elif formal_nvme:
        performance_note = "Explicitly marked as a >=7 GB/s local NVMe run by the caller."
    else:
        performance_note = "Functional/memory result only; do not use this node for NVMe claims."
    result: dict[str, object] = {
        "status": "runtime_error",
        "configuration": vars(args) | {"output": str(args.output)},
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "storage_backend": (
            "simulated_nvme"
            if args.storage == "simulated-nvme"
            else "nvme"
            if args.storage.startswith("nvme-")
            else "memory"
        ),
        "storage_performance_valid": formal_nvme,
        "storage_performance_note": performance_note,
    }
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("paged benchmark requires a visible CUDA GPU")
        dtype = getattr(torch, args.dtype)
        cu = make_bounds(args.tokens, args.segments)
        direct_io = not args.buffered_io_for_tests
        page_target_bytes = args.page_mib * 2**20
        q_page_target_bytes = (args.q_page_mib or args.page_mib) * 2**20
        kv_page_target_bytes = (args.kv_page_mib or args.page_mib) * 2**20
        preparation_seconds = 0.0
        quantization_seconds = 0.0
        simulated_device = None
        if args.storage in {"memory", "simulated-nvme"}:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            q = make_tensor((args.tokens, args.q_heads, args.head_dim), dtype, generator)
            k = make_tensor((args.tokens, args.kv_heads, args.head_dim), dtype, generator)
            v = make_tensor((args.tokens, args.kv_heads, args.head_dim), dtype, generator)
            q_memory_source = MemoryPageSource(
                q=q,
                cu_seqlens_q=cu,
                page_target_bytes=q_page_target_bytes,
                block_n=64,
            )
            kv_memory_source = MemoryPageSource(
                k=k,
                v=v,
                cu_seqlens_k=cu,
                page_target_bytes=kv_page_target_bytes,
                block_n=64,
            )
            storage_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
            direct_io = False
            if args.storage == "simulated-nvme":
                simulated_config = SimulatedNvmeConfig(
                    read_bandwidth_bytes_per_second=args.simulate_read_gbps * 1e9,
                    write_bandwidth_bytes_per_second=args.simulate_write_gbps * 1e9,
                    read_latency_seconds=args.simulate_read_latency_us * 1e-6,
                    write_latency_seconds=args.simulate_write_latency_us * 1e-6,
                    max_concurrent_reads=args.simulate_max_concurrent_reads,
                    max_concurrent_writes=args.simulate_max_concurrent_writes,
                    jitter_fraction=args.simulate_jitter_fraction,
                    random_seed=args.seed,
                )
                simulated_device = SimulatedNvmeDevice(simulated_config)
                q_source = SimulatedPageSource(q_memory_source, device=simulated_device)
                kv_source = SimulatedPageSource(kv_memory_source, device=simulated_device)
                result["simulated_nvme_config"] = simulated_config.as_dict()
            else:
                q_source = q_memory_source
                kv_source = kv_memory_source
        else:
            if (
                q_page_target_bytes != page_target_bytes
                or kv_page_target_bytes != page_target_bytes
            ):
                raise ValueError(
                    "separate Q/KV page sizes are currently supported only by memory and "
                    "simulated-nvme benchmarks"
                )
            if args.store_dir is None:
                temporary = tempfile.TemporaryDirectory(prefix="seqattn-bench-")
                store_dir = Path(temporary.name) / "qkv"
            else:
                store_dir = args.store_dir
            storage_dtype = (
                "int8"
                if args.storage == "nvme-int8"
                else ("bf16" if dtype == torch.bfloat16 else "fp16")
            )
            source, preparation_seconds, quantization_seconds = stream_store(
                store_dir,
                tokens=args.tokens,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                cu_seqlens=cu,
                page_target_bytes=page_target_bytes,
                kv_storage_dtype=storage_dtype,
                direct_io=direct_io,
                seed=args.seed,
            )
            q_source = source
            kv_source = source

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
            enable_nvtx=args.nvtx,
        )
        paged_config = PagedAttentionConfig(
            attention=attention_config,
            host_memory_budget_bytes=int(args.host_budget_gib * 2**30),
            pinned_staging_budget_bytes=1 * 2**30,
            direct_io_bounce_budget_bytes=512 * 2**20,
            metadata_margin_bytes=128 * 2**20,
            page_target_bytes=kv_page_target_bytes,
            io_workers=args.io_workers,
            io_queue_depth=args.queue_depth,
            num_output_buffers=args.num_output_buffers,
            direct_io=direct_io,
            kv_storage_dtype=storage_dtype,
        )
        runner = PagedAttentionRunner(paged_config, device="cuda")
        durations = []
        last_stats = None
        checksum = 0.0
        torch.cuda.reset_peak_memory_stats()
        with ProcessMemorySampler() as sampler:
            for repeat in range(args.repeats):
                profile_repeat = args.cuda_profiler_repeat == repeat
                stats = PagedAttentionStats()
                if args.storage in {"memory", "simulated-nvme"}:

                    def consume(_page, data):
                        nonlocal checksum
                        checksum += float(data[0, 0, 0])

                    sink = CallbackOutputSink(consume)
                    if simulated_device is not None:
                        sink = SimulatedPageSink(sink, device=simulated_device)
                else:
                    assert q_source.path is not None
                    output_dir = q_source.path.parent / f"output-{os.getpid()}-{repeat}"
                    sink = NvmeOutputSink(output_dir, direct_io=direct_io)
                if profile_repeat:
                    torch.cuda.cudart().cudaProfilerStart()
                started = time.perf_counter()
                try:
                    runner.run(q_source, kv_source, cu, cu, sink, causal=args.causal, stats=stats)
                    torch.cuda.synchronize()
                finally:
                    if profile_repeat:
                        torch.cuda.cudart().cudaProfilerStop()
                durations.append(time.perf_counter() - started)
                last_stats = stats
        assert last_stats is not None
        lengths = torch.diff(cu).tolist()
        flop = 4 * args.q_heads * args.head_dim * sum(length * length for length in lengths)
        mean_seconds = sum(durations) / len(durations)
        result.update(
            status="success",
            preparation_seconds=preparation_seconds,
            quantization_seconds=quantization_seconds,
            seconds=durations,
            mean_seconds=mean_seconds,
            tokens_per_second=args.tokens / mean_seconds,
            effective_tflops=flop / mean_seconds / 1e12,
            checksum=checksum,
            q_page_count=len(q_source.q_pages),
            kv_page_count=len(kv_source.kv_pages),
            process_peak_rss_bytes=sampler.peak_rss_bytes,
            nvml_process_peak_bytes=sampler.peak_vram_bytes,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            memory_samples=sampler.samples,
            paged_stats=last_stats.as_dict(),
        )
    except Exception as error:  # noqa: BLE001 - benchmark records failures as JSON
        result["status"] = (
            "oom" if isinstance(error, (MemoryError, torch.OutOfMemoryError)) else "runtime_error"
        )
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if temporary is not None:
            temporary.cleanup()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
