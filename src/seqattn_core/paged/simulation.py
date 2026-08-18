from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import torch

from .layout import PageDescriptor, PageReadMetrics, TensorLayout
from .memory_budget import HostMemoryPlan
from .protocols import PageReader, PageSink, PageSource, PageWriter


@dataclass(frozen=True)
class SimulatedNvmeConfig:
    """Timing model for an in-memory, aggregate-bandwidth-limited NVMe device."""

    read_bandwidth_bytes_per_second: float = 7e9
    write_bandwidth_bytes_per_second: float = 6e9
    read_latency_seconds: float = 80e-6
    write_latency_seconds: float = 100e-6
    max_concurrent_reads: int = 4
    max_concurrent_writes: int = 4
    jitter_fraction: float = 0.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("read_bandwidth_bytes_per_second", self.read_bandwidth_bytes_per_second),
            ("write_bandwidth_bytes_per_second", self.write_bandwidth_bytes_per_second),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("read_latency_seconds", self.read_latency_seconds),
            ("write_latency_seconds", self.write_latency_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_concurrent_reads <= 0 or self.max_concurrent_writes <= 0:
            raise ValueError("simulated NVMe concurrency limits must be positive")
        if not 0.0 <= self.jitter_fraction < 1.0:
            raise ValueError("jitter_fraction must be within [0, 1)")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedIoDelay:
    elapsed_seconds: float
    service_seconds: float
    queue_seconds: float
    physical_bytes: int


@dataclass(frozen=True)
class _TransferReservation:
    arrival_seconds: float
    latency_ready_seconds: float
    transfer_start_seconds: float
    finish_seconds: float
    service_seconds: float
    queue_seconds: float


class _BandwidthTimeline:
    """Serializes transfer time while allowing command latency to overlap."""

    def __init__(
        self,
        bandwidth_bytes_per_second: float,
        latency_seconds: float,
        jitter_fraction: float,
        random_seed: int,
    ) -> None:
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self.latency_seconds = latency_seconds
        self.jitter_fraction = jitter_fraction
        self._random = random.Random(random_seed)
        self._next_transfer_seconds = 0.0
        self._lock = threading.Lock()

    def reserve(self, physical_bytes: int, *, arrival_seconds: float) -> _TransferReservation:
        if physical_bytes < 0:
            raise ValueError("physical_bytes must be non-negative")
        with self._lock:
            jitter = 1.0
            if self.jitter_fraction:
                jitter += self._random.uniform(-self.jitter_fraction, self.jitter_fraction)
            latency = self.latency_seconds * jitter
            transfer = physical_bytes / self.bandwidth_bytes_per_second * jitter
            latency_ready = arrival_seconds + latency
            transfer_start = max(latency_ready, self._next_transfer_seconds)
            finish = transfer_start + transfer
            self._next_transfer_seconds = finish
        return _TransferReservation(
            arrival_seconds=arrival_seconds,
            latency_ready_seconds=latency_ready,
            transfer_start_seconds=transfer_start,
            finish_seconds=finish,
            service_seconds=latency + transfer,
            queue_seconds=max(0.0, transfer_start - latency_ready),
        )


class _DirectionThrottle:
    def __init__(
        self,
        timeline: _BandwidthTimeline,
        max_concurrent: int,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
    ) -> None:
        self.timeline = timeline
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._clock = clock
        self._sleeper = sleeper

    def wait(self, physical_bytes: int) -> SimulatedIoDelay:
        started = self._clock()
        self._slots.acquire()
        acquired = self._clock()
        try:
            reservation = self.timeline.reserve(physical_bytes, arrival_seconds=acquired)
            remaining = reservation.finish_seconds - self._clock()
            if remaining > 0:
                self._sleeper(remaining)
            elapsed = max(0.0, self._clock() - started)
            return SimulatedIoDelay(
                elapsed_seconds=elapsed,
                service_seconds=reservation.service_seconds,
                queue_seconds=(acquired - started) + reservation.queue_seconds,
                physical_bytes=physical_bytes,
            )
        finally:
            self._slots.release()


class SimulatedNvmeDevice:
    """Shared read/write timing state for simulated NVMe sources and sinks."""

    def __init__(
        self,
        config: SimulatedNvmeConfig | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = SimulatedNvmeConfig() if config is None else config
        self._read_timeline = _BandwidthTimeline(
            self.config.read_bandwidth_bytes_per_second,
            self.config.read_latency_seconds,
            self.config.jitter_fraction,
            self.config.random_seed,
        )
        self._write_timeline = _BandwidthTimeline(
            self.config.write_bandwidth_bytes_per_second,
            self.config.write_latency_seconds,
            self.config.jitter_fraction,
            self.config.random_seed + 1,
        )
        self._reads = _DirectionThrottle(
            self._read_timeline,
            self.config.max_concurrent_reads,
            clock=clock,
            sleeper=sleeper,
        )
        self._writes = _DirectionThrottle(
            self._write_timeline,
            self.config.max_concurrent_writes,
            clock=clock,
            sleeper=sleeper,
        )

    def throttle_read(self, physical_bytes: int) -> SimulatedIoDelay:
        return self._reads.wait(physical_bytes)

    def throttle_write(self, physical_bytes: int) -> SimulatedIoDelay:
        return self._writes.wait(physical_bytes)


def _with_delay(metrics: PageReadMetrics, delay: SimulatedIoDelay) -> PageReadMetrics:
    return replace(
        metrics,
        read_seconds=metrics.read_seconds + delay.elapsed_seconds,
        simulated_io_seconds=metrics.simulated_io_seconds + delay.elapsed_seconds,
        simulated_service_seconds=(metrics.simulated_service_seconds + delay.service_seconds),
        simulated_queue_seconds=metrics.simulated_queue_seconds + delay.queue_seconds,
        simulated_logical_bytes=metrics.simulated_logical_bytes + metrics.logical_bytes,
        simulated_physical_bytes=(metrics.simulated_physical_bytes + delay.physical_bytes),
    )


class _SimulatedPageReader(PageReader):
    def __init__(self, reader: PageReader, device: SimulatedNvmeDevice) -> None:
        self.reader = reader
        self.device = device

    def read_q(self, page: PageDescriptor, out: torch.Tensor) -> PageReadMetrics:
        metrics = self.reader.read_q(page, out)
        return _with_delay(metrics, self.device.throttle_read(metrics.physical_bytes))

    def read_kv(
        self,
        page: PageDescriptor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
        k_scales_out: torch.Tensor | None = None,
        v_scales_out: torch.Tensor | None = None,
    ) -> PageReadMetrics:
        metrics = self.reader.read_kv(page, k_out, v_out, k_scales_out, v_scales_out)
        return _with_delay(metrics, self.device.throttle_read(metrics.physical_bytes))

    def close(self) -> None:
        self.reader.close()


class SimulatedPageSource(PageSource):
    """Adds a shared NVMe timing model to any existing page source."""

    backing_kind = "simulated_nvme"
    direct_io = False

    def __init__(
        self,
        source: PageSource,
        config: SimulatedNvmeConfig | None = None,
        *,
        device: SimulatedNvmeDevice | None = None,
    ) -> None:
        if device is not None and config is not None:
            raise ValueError("pass either config or device, not both")
        self.source = source
        self.device = device or SimulatedNvmeDevice(config)
        self.q_layout = source.q_layout
        self.kv_layout = source.kv_layout
        self.q_pages = source.q_pages
        self.kv_pages = source.kv_pages
        self.cu_seqlens_q = source.cu_seqlens_q
        self.cu_seqlens_k = source.cu_seqlens_k

    def open_reader(self, memory_plan: HostMemoryPlan, queue_depth: int) -> PageReader:
        return _SimulatedPageReader(self.source.open_reader(memory_plan, queue_depth), self.device)


class _SimulatedPageWriter(PageWriter):
    def __init__(self, writer: PageWriter, device: SimulatedNvmeDevice) -> None:
        self.writer = writer
        self.device = device

    def write_page(self, page: PageDescriptor, data: torch.Tensor) -> PageReadMetrics:
        metrics = self.writer.write_page(page, data)
        return _with_delay(metrics, self.device.throttle_write(metrics.physical_bytes))

    def close(self) -> object:
        return self.writer.close()

    def abort(self) -> None:
        self.writer.abort()


class SimulatedPageSink(PageSink):
    """Adds a shared NVMe timing model to any existing page sink."""

    backing_kind = "simulated_nvme"
    direct_io = False

    def __init__(
        self,
        sink: PageSink,
        config: SimulatedNvmeConfig | None = None,
        *,
        device: SimulatedNvmeDevice | None = None,
    ) -> None:
        if device is not None and config is not None:
            raise ValueError("pass either config or device, not both")
        self.sink = sink
        self.device = device or SimulatedNvmeDevice(config)

    def open_writer(
        self,
        layout: TensorLayout,
        cu_seqlens: Sequence[int],
        pages: Sequence[PageDescriptor],
        memory_plan: HostMemoryPlan,
        queue_depth: int,
    ) -> PageWriter:
        writer = self.sink.open_writer(layout, cu_seqlens, pages, memory_plan, queue_depth)
        return _SimulatedPageWriter(writer, self.device)


__all__ = [
    "SimulatedIoDelay",
    "SimulatedNvmeConfig",
    "SimulatedNvmeDevice",
    "SimulatedPageSink",
    "SimulatedPageSource",
]
