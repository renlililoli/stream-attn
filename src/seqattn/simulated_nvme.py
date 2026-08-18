"""Compatibility facade for the in-memory NVMe timing model."""

from seqattn_core.paged.simulation import (
    SimulatedIoDelay,
    SimulatedNvmeConfig,
    SimulatedNvmeDevice,
    SimulatedPageSink,
    SimulatedPageSource,
)
from seqattn_core.paged.simulation import _BandwidthTimeline

__all__ = [
    "SimulatedIoDelay",
    "SimulatedNvmeConfig",
    "SimulatedNvmeDevice",
    "SimulatedPageSink",
    "SimulatedPageSource",
]
