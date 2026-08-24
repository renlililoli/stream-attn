from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover - optional benchmark dependency
    pynvml = None


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def make_bounds(tokens: int, segments: int) -> torch.Tensor:
    if segments <= 0 or segments > tokens:
        raise ValueError("segments must be within [1, tokens]")
    base, remainder = divmod(tokens, segments)
    lengths = [base + (index < remainder) for index in range(segments)]
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32)


def make_host_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator
) -> torch.Tensor:
    # Fill pageable memory first so the OpenMP random kernels fault pages in
    # parallel, then register with the CUDA driver. Allocating with
    # pin_memory=True up front faults the pages during the fill and observes
    # single-core throughput for multi-GB buffers.
    tensor = torch.randn(shape, dtype=dtype, generator=generator)
    return tensor.pin_memory()


def make_pinned_host_tensors_parallel(
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    *,
    workers: int,
) -> tuple[torch.Tensor, ...]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not shapes:
        return ()

    def allocate(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    allocation_workers = min(workers, len(shapes))
    with ThreadPoolExecutor(
        max_workers=allocation_workers, thread_name_prefix="seqattn-alloc"
    ) as pool:
        return tuple(pool.map(allocate, shapes))


def make_host_tensors_parallel(
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
    *,
    seed: int,
    workers: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, ...]:
    if workers <= 0 or chunk_tokens <= 0:
        raise ValueError("workers and chunk_tokens must be positive")
    tensors = make_pinned_host_tensors_parallel(shapes, dtype, workers=workers)
    tasks = [
        (tensor_index, chunk_index, start, min(start + chunk_tokens, tensor.shape[0]))
        for tensor_index, tensor in enumerate(tensors)
        for chunk_index, start in enumerate(range(0, tensor.shape[0], chunk_tokens))
    ]

    def fill(task: tuple[int, int, int, int]) -> None:
        tensor_index, chunk_index, start, stop = task
        generator = torch.Generator(device="cpu").manual_seed(
            seed + tensor_index * 1_000_003 + chunk_index
        )
        tensors[tensor_index][start:stop].normal_(generator=generator)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seqattn-data") as pool:
        list(pool.map(fill, tasks))
    return tensors


def process_rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


_nvml_handle = None


def process_vram_bytes(pid: int) -> int:
    global _nvml_handle
    if pynvml is None or not torch.cuda.is_available():
        return 0
    if _nvml_handle is None:
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle)
    return sum(
        int(process.usedGpuMemory)
        for process in processes
        if process.pid == pid and process.usedGpuMemory is not None
    )


def process_vram_mib(pid: int) -> float:
    return process_vram_bytes(pid) / 2**20


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_mib = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, process_vram_mib(self.pid))
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak_mib = max(self.peak_mib, process_vram_mib(self.pid))


class ProcessMemorySampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.pid = os.getpid()
        self.peak_rss_bytes = 0
        self.peak_vram_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, process_rss_bytes())
            self.peak_vram_bytes = max(self.peak_vram_bytes, process_vram_bytes(self.pid))
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_rss_bytes = max(self.peak_rss_bytes, process_rss_bytes())
        self.peak_vram_bytes = max(self.peak_vram_bytes, process_vram_bytes(self.pid))


def configure_allocator(target_vram_mib: int | None, safety_mib: int) -> dict[str, float]:
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


__all__ = [
    "MemorySampler",
    "ProcessMemorySampler",
    "atomic_json",
    "configure_allocator",
    "make_bounds",
    "make_host_tensor",
    "make_host_tensors_parallel",
    "make_pinned_host_tensors_parallel",
    "process_rss_bytes",
    "process_vram_bytes",
    "process_vram_mib",
]
