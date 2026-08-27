from __future__ import annotations

from typing import Protocol

import torch

from .tasks import QueryTask


class DeviceOutputTransform(Protocol):
    """Transforms a finalized tile on the runner device and compute stream.

    The callback must enqueue all work on the current stream and return a
    fully written tensor on the planned CUDA device covering the complete
    destination range ``[start, stop)``.
    """

    def __call__(
        self,
        attention: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor: ...


class DeviceOutputConsumer(Protocol):
    """Consumes finalized attention tiles on the runner compute stream.

    ``__call__`` and ``finish`` run with the planned CUDA device selected and
    the runner compute stream current. All work that reads the supplied tile
    must be submitted to that stream before the callback returns. A consumer
    may use additional streams only after establishing explicit CUDA event
    dependencies from the compute stream.

    ``synchronize`` must not return until every destination range passed to
    ``__call__`` has been completely written and is safe for the caller to
    read. The consumer owns validation and lifetime management for its final
    destination.
    """

    def __call__(self, attention: torch.Tensor, start: int, stop: int) -> None: ...

    def finish(self) -> None: ...

    def synchronize(self) -> None: ...


class TaskDeviceOutputConsumer(DeviceOutputConsumer, Protocol):
    """Device consumer with an explicit per-query-task completion contract.

    ``begin_task`` is called before a single ``QueryTask`` is executed.
    ``finish_task`` must return an event that becomes complete only after the
    task's destination range is fully written. Timing accessors are read after
    that event has synchronized and must describe only the current task.
    """

    def begin_task(self, task: QueryTask) -> None: ...

    def finish_task(self) -> torch.cuda.Event: ...

    def task_d2h_seconds(self) -> float: ...

    def task_d2h_bytes(self) -> int: ...


__all__ = ["DeviceOutputConsumer", "DeviceOutputTransform", "TaskDeviceOutputConsumer"]
