from __future__ import annotations

from dataclasses import asdict, dataclass, field

from seqattn_core._plugin_api import H3DiTStats, ProjectedAttentionStats, StreamingAttentionStats


@dataclass
class DynamicDeviceStats:
    task_count: int = 0
    q_tokens: int = 0
    q_tokens_min: int = 0
    q_tokens_max: int = 0
    q_tokens_average: float = 0.0
    busy_seconds: float = 0.0
    attention_seconds: float = 0.0
    h2d_seconds: float = 0.0
    d2h_seconds: float = 0.0
    attention_flops: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    effective_tflops: float = 0.0
    h2d_gbps: float = 0.0
    d2h_gbps: float = 0.0
    effective_tflops_ema: float = 0.0
    h2d_gbps_ema: float = 0.0
    d2h_gbps_ema: float = 0.0
    task_elapsed_ema: float = 0.0
    q_current: int = 0
    work_fraction: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicTaskTrace:
    device: str
    segment_id: int
    q_start: int
    q_stop: int
    claim_order: int
    requested_q: int
    actual_q: int
    h2d_seconds: float
    attention_seconds: float
    consumer_seconds: float
    d2h_seconds: float
    elapsed_seconds: float
    q_before: int
    q_after: int
    segment_clamped: bool
    tail_clamped: bool


@dataclass
class MultiGpuAttentionStats:
    wall_seconds: float = 0.0
    per_device: dict[str, StreamingAttentionStats] = field(default_factory=dict)
    dynamic_per_device: dict[str, DynamicDeviceStats] = field(default_factory=dict)
    task_trace: list[DynamicTaskTrace] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "per_device": {
                device: device_stats.as_dict() for device, device_stats in self.per_device.items()
            },
            "dynamic_per_device": {
                device: device_stats.as_dict()
                for device, device_stats in self.dynamic_per_device.items()
            },
            "task_trace": [asdict(item) for item in self.task_trace],
        }


@dataclass
class MultiGpuH3DiTStats:
    wall_seconds: float = 0.0
    blocks: int = 0
    final_hidden_d2h_bytes: int = 0
    post_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes_peak: int = 0
    per_device_estimated_workspace_bytes: dict[str, int] = field(default_factory=dict)
    projection: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)
    attention: MultiGpuAttentionStats = field(default_factory=MultiGpuAttentionStats)
    per_device: dict[str, H3DiTStats] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "DynamicDeviceStats",
    "DynamicTaskTrace",
    "MultiGpuAttentionStats",
    "MultiGpuH3DiTStats",
]
