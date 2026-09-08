"""Offline, device-neutral performance and physical-allocation estimation."""

from .h3 import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3OperatorMeasurement,
    H3OperatorProfile,
    H3OperatorSample,
    H3ScratchBuffer,
    H3WeightPolicy,
    build_h3_block_execution,
    measure_cuda_h3_operator,
)
from .memory import MemoryPoint, PoolMemoryStats, TimelineStats, memory_statistics
from .report import timeline_report_data, write_timeline_report
from .schedule import schedule_execution, trace_from_measurements
from .search import ActivationMemoryEstimate, CandidateEstimate, estimate_activation_memory
from .specs import (
    BufferLifetime,
    BufferSpec,
    ExecutionSpec,
    ExecutionTrace,
    MemoryPool,
    OperationEvent,
    OperationSpec,
    RateProfile,
)

__all__ = [
    "ActivationMemoryEstimate",
    "BufferLifetime",
    "BufferSpec",
    "CandidateEstimate",
    "ExecutionSpec",
    "ExecutionTrace",
    "H3BlockShape",
    "H3CallbackConfig",
    "H3DeviceProfile",
    "H3ExecutionConfig",
    "H3OperatorMeasurement",
    "H3OperatorProfile",
    "H3OperatorSample",
    "H3ScratchBuffer",
    "H3WeightPolicy",
    "MemoryPoint",
    "MemoryPool",
    "OperationEvent",
    "OperationSpec",
    "PoolMemoryStats",
    "RateProfile",
    "TimelineStats",
    "build_h3_block_execution",
    "estimate_activation_memory",
    "measure_cuda_h3_operator",
    "memory_statistics",
    "schedule_execution",
    "timeline_report_data",
    "trace_from_measurements",
    "write_timeline_report",
]
