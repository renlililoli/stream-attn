from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StreamingAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    compute_pipeline_seconds: float = 0.0
    q_chunks: int = 0
    kv_tiles: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    estimated_workspace_bytes: int = 0
    q_chunk_tokens: int = 0
    kv_chunk_tokens: int = 0
    max_resident_q_tokens: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


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
class ProjectedAttentionStats:
    backend: str = ""
    wall_seconds: float = 0.0
    projection_seconds: float = 0.0
    attention_output_seconds: float = 0.0
    projection_chunks: int = 0
    projection_tokens: int = 0
    projection_hidden_h2d_bytes: int = 0
    projection_qkv_d2h_bytes: int = 0
    raw_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes: int = 0
    attention: StreamingAttentionStats = field(default_factory=StreamingAttentionStats)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["total_h2d_bytes"] = self.projection_hidden_h2d_bytes + self.attention.h2d_bytes
        result["total_d2h_bytes"] = self.projection_qkv_d2h_bytes + self.attention.d2h_bytes
        return result


@dataclass
class H3DiTStats:
    backend: str = ""
    wall_seconds: float = 0.0
    blocks: int = 0
    mlp_chunks: int = 0
    mlp_cross_q_boundaries: int = 0
    final_hidden_d2h_bytes: int = 0
    post_attention_roundtrip_bytes_avoided: int = 0
    qkv_host_bytes_peak: int = 0
    estimated_workspace_bytes: int = 0
    projection: ProjectedAttentionStats = field(default_factory=ProjectedAttentionStats)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


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


@dataclass
class PagedAttentionStats:
    backend: str = ""
    kv_storage_dtype: str = ""
    wall_seconds: float = 0.0
    nvme_read_seconds: float = 0.0
    nvme_write_seconds: float = 0.0
    simulated_read_seconds: float = 0.0
    simulated_write_seconds: float = 0.0
    simulated_read_service_seconds: float = 0.0
    simulated_write_service_seconds: float = 0.0
    simulated_read_queue_seconds: float = 0.0
    simulated_write_queue_seconds: float = 0.0
    io_queue_wait_seconds: float = 0.0
    cache_lookup_seconds: float = 0.0
    pinned_copy_seconds: float = 0.0
    gpu_kernel_seconds: float = 0.0
    output_write_seconds: float = 0.0
    quantization_seconds: float = 0.0
    q_pages: int = 0
    q_chunks: int = 0
    kv_page_scans: int = 0
    kv_pages: int = 0
    state_spills: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: float = 0.0
    nvme_logical_read_bytes: int = 0
    nvme_physical_read_bytes: int = 0
    nvme_logical_write_bytes: int = 0
    nvme_physical_write_bytes: int = 0
    simulated_logical_read_bytes: int = 0
    simulated_physical_read_bytes: int = 0
    simulated_logical_write_bytes: int = 0
    simulated_physical_write_bytes: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    q_chunk_tokens: int = 0
    kv_chunk_tokens: int = 0
    estimated_workspace_bytes: int = 0
    operator_host_allocated_bytes: int = 0
    operator_host_peak_bytes: int = 0
    pinned_peak_bytes: int = 0
    direct_io_bounce_peak_bytes: int = 0
    dram_cache_peak_bytes: int = 0
    host_memory_budget_bytes: int = 0

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)
