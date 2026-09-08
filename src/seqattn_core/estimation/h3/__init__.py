"""Complete H3 dense runner/callback modeling on device-neutral resources."""

from .calibration import H3OperatorMeasurement, measure_cuda_h3_operator
from .pipeline import build_h3_block_execution
from .profiles import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3OperatorProfile,
    H3OperatorSample,
    H3ScratchBuffer,
    H3WeightPolicy,
)

__all__ = [
    "H3BlockShape",
    "H3CallbackConfig",
    "H3DeviceProfile",
    "H3ExecutionConfig",
    "H3OperatorMeasurement",
    "H3OperatorProfile",
    "H3OperatorSample",
    "H3ScratchBuffer",
    "H3WeightPolicy",
    "build_h3_block_execution",
    "measure_cuda_h3_operator",
]
