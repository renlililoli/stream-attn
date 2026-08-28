from seqattn_core._plugin_api import PLUGIN_API_VERSION, QueryTaskMeasurement

if PLUGIN_API_VERSION != "0.3.0a4":
    raise ImportError(
        "seqattn-multigpu 0.3.0a4 requires the seqattn-core 0.3.0a4 plugin API; "
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
from .projection import DynamicQKVProjectionCursor, MultiGpuQKVProjectionRunner
from .stats import (
    DynamicDeviceStats,
    DynamicTaskTrace,
    MultiGpuAttentionStats,
    MultiGpuH3DiTStats,
)
from .streaming import (
    DeviceQuerySchedule,
    MultiGpuAttentionPlan,
    MultiGpuDeviceSpec,
    MultiGpuStreamingAttentionRunner,
    build_multi_gpu_plan,
)

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

__version__ = "0.3.0a4"
