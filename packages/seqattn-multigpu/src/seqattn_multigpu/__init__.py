from seqattn_core._plugin_api import PLUGIN_API_VERSION, QueryTaskMeasurement

if PLUGIN_API_VERSION != "0.4.0a1":
    raise ImportError(
        "seqattn-multigpu 0.4.0a1 requires the seqattn-core 0.4.0a1 plugin API; "
        f"found {PLUGIN_API_VERSION}"
    )

from .dit import MultiGpuH3MaterializedRunner
from .dynamic import (
    DynamicControllerSnapshot,
    DynamicQController,
    DynamicQueryCursor,
    DynamicScheduleConfig,
    DynamicWorkloadSignature,
)
from .planning import (
    DeviceQuerySchedule,
    MultiGpuAttentionPlan,
    MultiGpuDeviceSpec,
    build_multi_gpu_plan,
)
from .projection import DynamicQKVProjectionCursor, MultiGpuQKVProjectionRunner
from .stats import (
    DynamicDeviceStats,
    DynamicTaskTrace,
    MultiGpuAttentionStats,
    MultiGpuH3DiTStats,
)
from .streaming import MultiGpuStreamingAttentionRunner

__all__ = [
    "DeviceQuerySchedule",
    "DynamicControllerSnapshot",
    "DynamicDeviceStats",
    "DynamicQController",
    "DynamicQKVProjectionCursor",
    "DynamicQueryCursor",
    "DynamicScheduleConfig",
    "DynamicTaskTrace",
    "DynamicWorkloadSignature",
    "MultiGpuAttentionPlan",
    "MultiGpuAttentionStats",
    "MultiGpuDeviceSpec",
    "MultiGpuH3DiTStats",
    "MultiGpuH3MaterializedRunner",
    "MultiGpuQKVProjectionRunner",
    "MultiGpuStreamingAttentionRunner",
    "QueryTaskMeasurement",
    "build_multi_gpu_plan",
]

__version__ = "0.4.0a1"
