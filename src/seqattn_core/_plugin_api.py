"""Versioned private bridge for separately distributed SeqAttn plugins."""

from .config import StreamingAttentionConfig
from .dit.consumer import H3DeviceOutputConsumer
from .dit.types import (
    H3BlockOps,
    H3MaterializedProjection,
    H3SequenceMeta,
    estimate_h3_consumer_workspace_bytes,
)
from .dit.workspace import H3BlockWorkspace
from .planner import AttentionPlan, build_plan
from .projection import ProjectedAttentionRunner
from .projection.validation import validate_projected_qkv, validate_projection_hidden
from .projection.workspace import ProjectionWorkspace
from .stats import H3DiTStats, ProjectedAttentionStats, StreamingAttentionStats
from .streaming.measurement import QueryTaskMeasurement
from .streaming.protocols import DeviceOutputConsumer, TaskDeviceOutputConsumer
from .streaming.runner import StreamingAttentionRunner
from .streaming.tasks import QueryTask, build_query_tasks
from .validation import validate_host_qkv

PLUGIN_API_VERSION = "0.3.0a4"

__all__ = [
    "PLUGIN_API_VERSION",
    "AttentionPlan",
    "DeviceOutputConsumer",
    "H3BlockOps",
    "H3BlockWorkspace",
    "H3DeviceOutputConsumer",
    "H3DiTStats",
    "H3MaterializedProjection",
    "H3SequenceMeta",
    "ProjectedAttentionRunner",
    "ProjectedAttentionStats",
    "ProjectionWorkspace",
    "QueryTask",
    "QueryTaskMeasurement",
    "StreamingAttentionConfig",
    "StreamingAttentionRunner",
    "StreamingAttentionStats",
    "TaskDeviceOutputConsumer",
    "build_plan",
    "build_query_tasks",
    "estimate_h3_consumer_workspace_bytes",
    "validate_host_qkv",
    "validate_projected_qkv",
    "validate_projection_hidden",
]
