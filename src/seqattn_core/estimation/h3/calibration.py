"""Opt-in CUDA operator calibration; device-neutral estimation does not call this."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..specs import positive_int
from .profiles import H3OperatorSample


@dataclass(frozen=True)
class H3OperatorMeasurement:
    sample: H3OperatorSample
    seconds: tuple[float, ...]
    extra_workspace_bytes: tuple[int, ...]
    device_name: str
    torch_version: str
    provenance: str


def measure_cuda_h3_operator(
    operation,
    *,
    tokens: int,
    output_allocation_bytes: int,
    kv_tokens: int = 0,
    device="cuda",
    warmup: int = 2,
    repeats: int = 5,
    provenance: str,
    work: float | None = None,
) -> H3OperatorMeasurement:
    """Measure a caller-provided inference operator on the current CUDA stream.

    Inputs, weights and persistent workspaces must already exist; the callback
    must join any internal auxiliary streams before returning to this stream.
    Declare the new physical output storage (zero for in-place/direct-write
    output). Views sharing a backing allocation count once. Private workspace
    is measured as peak delta minus that output storage, not guessed from TFLOPS.

    Run in an otherwise idle process/device: this synchronizes CUDA and resets
    the process's allocator peak counters. Operators retaining newly allocated
    tensors after their result is released must declare them as persistent first.
    Timing uses CUDA events, not Nsight. The returned scalar uses median latency
    and maximum observed workspace, retaining raw repeats for inspection.
    """
    import torch

    for name, value in (("tokens", tokens), ("repeats", repeats)):
        positive_int(name, value)
    for name, value in (
        ("warmup", warmup),
        ("kv_tokens", kv_tokens),
        ("output_allocation_bytes", output_allocation_bytes),
    ):
        positive_int(name, value, allow_zero=True)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA operator calibration requires a CUDA device")
    durations, workspaces = [], []
    with torch.cuda.device(device), torch.inference_mode():
        for _ in range(warmup):
            result = operation()
            torch.cuda.synchronize(device)
            del result
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start.record()
            result = operation()
            end.record()
            end.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            if output_allocation_bytes > peak - baseline:
                raise ValueError(
                    "declared new output storage exceeds the observed allocation delta"
                )
            durations.append(start.elapsed_time(end) / 1000)
            workspaces.append(max(0, peak - baseline - output_allocation_bytes))
            del result
            torch.cuda.synchronize(device)
            if torch.cuda.memory_allocated(device) > baseline:
                raise ValueError(
                    "operator retained new allocations; declare persistent storage before calibration"
                )
    return H3OperatorMeasurement(
        H3OperatorSample(tokens, statistics.median(durations), max(workspaces), kv_tokens, work),
        tuple(durations),
        tuple(workspaces),
        torch.cuda.get_device_name(device),
        torch.__version__,
        provenance,
    )
