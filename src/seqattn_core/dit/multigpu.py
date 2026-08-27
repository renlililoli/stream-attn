from __future__ import annotations

import time
from collections.abc import Iterable, Mapping

import torch

from ..projection import ProjectedAttentionRunner
from ..projection.validation import validate_projection_hidden
from ..stats import H3DiTStats, MultiGpuH3DiTStats
from ..streaming import (
    DynamicScheduleConfig,
    MultiGpuAttentionPlan,
    MultiGpuStreamingAttentionRunner,
)
from .consumer import H3DeviceOutputConsumer
from .projection import MultiGpuQKVProjectionRunner
from .types import (
    H3BlockOps,
    H3MaterializedProjection,
    H3SequenceMeta,
    estimate_h3_consumer_workspace_bytes,
)
from .workspace import H3BlockWorkspace


def _normalize_device_values(
    value: int | Mapping[torch.device | str, int],
    *,
    devices: tuple[torch.device, ...],
    name: str,
) -> dict[str, int]:
    if isinstance(value, int):
        values = {str(device): value for device in devices}
    else:
        values = {str(torch.device(device)): item for device, item in value.items()}
        expected = {str(device) for device in devices}
        if set(values) != expected:
            raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    if any(item <= 0 for item in values.values()):
        raise ValueError(f"{name} values must be positive")
    return values


class MultiGpuH3MaterializedRunner:
    """Dynamic multi-GPU materialized-QKV and fused H3 consumer pipeline."""

    def __init__(
        self,
        projected_attention: ProjectedAttentionRunner,
        attention_plan: MultiGpuAttentionPlan,
        *,
        hidden_features: int,
        projection_chunk_tokens: int = 4096,
        num_final_output_buffers: int | Mapping[torch.device | str, int] = 2,
        dynamic_config: DynamicScheduleConfig | None = None,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if projection_chunk_tokens <= 0:
            raise ValueError("projection_chunk_tokens must be positive")
        if attention_plan.output_mode != "device_consumer":
            raise ValueError("the multi-GPU H3 runner requires device_consumer plans")
        if projected_attention.attention.backend != "triton":
            raise ValueError("the multi-GPU H3 runner requires the Triton backend")
        if projected_attention.plan.output_mode != "device_consumer":
            raise ValueError("the projected producer requires device_consumer output mode")
        if (
            projected_attention.plan.q_heads != attention_plan.q_heads
            or projected_attention.plan.kv_heads != attention_plan.kv_heads
            or projected_attention.plan.head_dim != attention_plan.head_dim
            or projected_attention.plan.dtype != attention_plan.dtype
            or projected_attention.plan.max_q_tokens != attention_plan.max_q_tokens
            or projected_attention.plan.max_kv_tokens != attention_plan.max_kv_tokens
        ):
            raise ValueError("projected and multi-GPU attention shapes must match")
        primary_device = projected_attention.plan.device
        primary_schedule = next(
            (
                schedule
                for schedule in attention_plan.schedules
                if schedule.device == primary_device
            ),
            None,
        )
        if primary_schedule is None:
            raise ValueError("the projected producer device must be in the multi-GPU plan")
        if primary_schedule.attention_plan != projected_attention.plan:
            raise ValueError("the primary multi-GPU plan must match the projected attention plan")

        devices = attention_plan.devices
        output_buffers = _normalize_device_values(
            num_final_output_buffers,
            devices=devices,
            name="num_final_output_buffers",
        )
        if any(value not in {1, 2} for value in output_buffers.values()):
            raise ValueError("num_final_output_buffers values must be 1 or 2")

        self.projected_attention = projected_attention
        self.attention_plan = attention_plan
        self.hidden_features = hidden_features
        self.primary_device = primary_device
        self.projection = MultiGpuQKVProjectionRunner(
            projected_attention,
            attention_plan,
            hidden_features=hidden_features,
            chunk_tokens=projection_chunk_tokens,
        )
        try:
            self.attention = MultiGpuStreamingAttentionRunner(
                attention_plan,
                runner_overrides={primary_device: projected_attention.attention},
                schedule_mode="dynamic",
                dynamic_config=dynamic_config,
            )
        except Exception:
            self.projection.close()
            raise
        self.workspaces = {}
        self.consumers = {}
        self.per_device_estimated_workspace_bytes = {}
        for schedule in attention_plan.schedules:
            device = str(schedule.device)
            workspace = H3BlockWorkspace(
                hidden_features=hidden_features,
                mlp_chunk_tokens=1,
                dtype=attention_plan.dtype,
                device=schedule.device,
                num_final_output_buffers=output_buffers[device],
                final_output_chunk_tokens=schedule.q_capacity_tokens,
            )
            self.workspaces[device] = workspace
            self.consumers[device] = H3DeviceOutputConsumer(workspace)
            consumer_bytes = estimate_h3_consumer_workspace_bytes(
                hidden_features=hidden_features,
                dtype=attention_plan.dtype,
                mlp_chunk_tokens=1,
                num_final_output_buffers=output_buffers[device],
                final_output_chunk_tokens=schedule.q_capacity_tokens,
            )
            self.per_device_estimated_workspace_bytes[device] = (
                schedule.attention_plan.estimated_workspace_bytes
                + consumer_bytes
                + self.projection.estimated_workspace_bytes[device]
            )

    def close(self) -> None:
        self.projection.close()
        self.attention.close()

    def _normalize_ops(
        self,
        ops_by_device: Mapping[torch.device | str, H3BlockOps],
    ) -> dict[str, H3BlockOps]:
        ops = {str(torch.device(device)): item for device, item in ops_by_device.items()}
        expected = {str(device) for device in self.attention_plan.devices}
        if set(ops) != expected:
            raise ValueError(f"ops_by_device must contain exactly {sorted(expected)}")
        return ops

    def _normalize_projections(
        self,
        projections_by_device: Mapping[torch.device | str, H3MaterializedProjection],
    ) -> dict[str, H3MaterializedProjection]:
        projections = {
            str(torch.device(device)): item for device, item in projections_by_device.items()
        }
        expected = {str(device) for device in self.attention_plan.devices}
        if set(projections) != expected:
            raise ValueError(f"projections_by_device must contain exactly {sorted(expected)}")
        return projections

    def _validate_inputs(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
    ) -> None:
        validate_projection_hidden(
            hidden_host,
            plan=self.projected_attention.plan,
            require_pinned=self.projected_attention.pipeline_config.require_pinned_hidden,
            hidden_features=self.hidden_features,
            name="hidden_host",
        )
        if hidden_host.shape[0] != self.attention_plan.max_q_tokens:
            raise ValueError("hidden_host token count must match the multi-GPU plan")
        sequence_meta.validate(hidden_host.shape[0])
        bounds = tuple(sequence_meta.cu_seqlens.to(dtype=torch.int64).tolist())
        if bounds != self.attention_plan.q_bounds or bounds != self.attention_plan.k_bounds:
            raise ValueError("sequence_meta does not match the multi-GPU self-attention plan")

    @torch.inference_mode()
    def run_block_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        projections_by_device: Mapping[torch.device | str, H3MaterializedProjection],
        ops_by_device: Mapping[torch.device | str, H3BlockOps],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: MultiGpuH3DiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_host, sequence_meta)
        projections = self._normalize_projections(projections_by_device)
        ops = self._normalize_ops(ops_by_device)
        stats = MultiGpuH3DiTStats() if stats is None else stats
        stats.per_device_estimated_workspace_bytes = dict(self.per_device_estimated_workspace_bytes)
        started = time.perf_counter()

        for schedule in self.attention_plan.schedules:
            device = str(schedule.device)
            device_stats = stats.per_device.setdefault(device, H3DiTStats())
            device_stats.backend = "triton"
            device_stats.qkv_storage_policy = "materialized"
            device_stats.estimated_workspace_bytes = self.per_device_estimated_workspace_bytes[
                device
            ]

        q_cpu, k_cpu, v_cpu = self.projection.run(
            hidden_host,
            projections,
            stats=stats.projection,
            per_device_stats={
                str(schedule.device): stats.per_device[str(schedule.device)].projection
                for schedule in self.attention_plan.schedules
            },
        )
        qkv_host_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
        )
        stats.qkv_host_bytes_peak = max(stats.qkv_host_bytes_peak, qkv_host_bytes)
        raw_attention_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.projection.raw_attention_roundtrip_bytes_avoided += 2 * raw_attention_bytes

        for schedule in self.attention_plan.schedules:
            device = str(schedule.device)
            self.consumers[device].reset(
                destination_hidden_host=hidden_host,
                residual_hidden_host=hidden_host,
                ops=ops[device],
                stats=stats.per_device[device],
            )

        attention_started = time.perf_counter()
        self.attention.run_with_device_consumers(
            q_cpu,
            k_cpu,
            v_cpu,
            sequence_meta.cu_seqlens,
            sequence_meta.cu_seqlens,
            output_consumers=self.consumers,
            device_contexts={device: item.consumer_context for device, item in ops.items()},
            softmax_scale=softmax_scale,
            causal=causal,
            stats=stats.attention,
        )
        stats.projection.attention_output_seconds += time.perf_counter() - attention_started

        element_size = hidden_host.element_size()
        for schedule in self.attention_plan.schedules:
            device = str(schedule.device)
            device_stats = stats.per_device[device]
            q_tokens = stats.attention.dynamic_per_device[device].q_tokens
            shard_bytes = q_tokens * self.hidden_features * element_size
            device_stats.post_attention_roundtrip_bytes_avoided += 2 * shard_bytes
            device_stats.blocks += 1
        hidden_bytes = hidden_host.numel() * element_size
        stats.final_hidden_d2h_bytes += hidden_bytes
        stats.post_attention_roundtrip_bytes_avoided += 2 * hidden_bytes
        stats.blocks += 1
        elapsed = time.perf_counter() - started
        stats.wall_seconds += elapsed
        stats.projection.wall_seconds += elapsed
        return hidden_host

    @torch.inference_mode()
    def run_blocks_(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
        blocks: Iterable[
            tuple[
                Mapping[torch.device | str, H3MaterializedProjection],
                Mapping[torch.device | str, H3BlockOps],
            ]
        ],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: MultiGpuH3DiTStats | None = None,
    ) -> torch.Tensor:
        stats = MultiGpuH3DiTStats() if stats is None else stats
        for projections_by_device, ops_by_device in blocks:
            self.run_block_(
                hidden_host,
                sequence_meta,
                projections_by_device,
                ops_by_device,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )
        return hidden_host


MultiGpuH3DiTRunner = MultiGpuH3MaterializedRunner


__all__ = ["MultiGpuH3DiTRunner", "MultiGpuH3MaterializedRunner"]
