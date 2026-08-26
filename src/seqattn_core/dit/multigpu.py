from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Literal

import torch

from ..projection import ProjectedAttentionRunner
from ..stats import H3DiTStats, MultiGpuH3DiTStats
from ..streaming import (
    DynamicScheduleConfig,
    MultiGpuAttentionPlan,
    MultiGpuStreamingAttentionRunner,
)
from .consumer import H3DeviceOutputConsumer
from .types import H3BlockOps, H3SequenceMeta, estimate_h3_consumer_workspace_bytes
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


class MultiGpuH3DiTRunner:
    """Static or dynamic heterogeneous Q scheduling for the fused H3 block path."""

    def __init__(
        self,
        projected_attention: ProjectedAttentionRunner,
        attention_plan: MultiGpuAttentionPlan,
        *,
        hidden_features: int,
        mlp_chunk_tokens: int | Mapping[torch.device | str, int] | None = None,
        num_final_output_buffers: int | Mapping[torch.device | str, int] = 2,
        schedule_mode: Literal["static", "dynamic"] | None = None,
        dynamic_config: DynamicScheduleConfig | None = None,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
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
        resolved_schedule_mode = (
            attention_plan.schedule_mode if schedule_mode is None else schedule_mode
        )
        if resolved_schedule_mode == "dynamic":
            mlp_chunks = {str(device): 1 for device in devices}
        else:
            if mlp_chunk_tokens is None:
                raise ValueError("mlp_chunk_tokens is required for static H3 scheduling")
            mlp_chunks = _normalize_device_values(
                mlp_chunk_tokens,
                devices=devices,
                name="mlp_chunk_tokens",
            )
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
        self.attention = MultiGpuStreamingAttentionRunner(
            attention_plan,
            runner_overrides={primary_device: projected_attention.attention},
            schedule_mode=schedule_mode,
            dynamic_config=dynamic_config,
        )
        self.schedule_mode = self.attention.schedule_mode
        self.workspaces = {}
        self.consumers = {}
        self.per_device_estimated_workspace_bytes = {}
        element_size = torch.empty((), dtype=attention_plan.dtype).element_size()
        projection_bytes = (
            projected_attention.pipeline_config.num_projection_buffers
            * projected_attention.pipeline_config.projection_chunk_tokens
            * hidden_features
            * element_size
        )
        for schedule in attention_plan.schedules:
            device = str(schedule.device)
            final_output_chunk_tokens = (
                mlp_chunks[device] if self.schedule_mode == "static" else schedule.q_capacity_tokens
            )
            workspace = H3BlockWorkspace(
                hidden_features=hidden_features,
                mlp_chunk_tokens=mlp_chunks[device],
                dtype=attention_plan.dtype,
                device=schedule.device,
                num_final_output_buffers=output_buffers[device],
                final_output_chunk_tokens=final_output_chunk_tokens,
            )
            self.workspaces[device] = workspace
            self.consumers[device] = H3DeviceOutputConsumer(workspace)
            consumer_bytes = estimate_h3_consumer_workspace_bytes(
                hidden_features=hidden_features,
                dtype=attention_plan.dtype,
                mlp_chunk_tokens=mlp_chunks[device],
                num_final_output_buffers=output_buffers[device],
                final_output_chunk_tokens=final_output_chunk_tokens,
            )
            self.per_device_estimated_workspace_bytes[device] = (
                schedule.attention_plan.estimated_workspace_bytes
                + consumer_bytes
                + (projection_bytes if schedule.device == primary_device else 0)
            )

    def close(self) -> None:
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

    def _validate_inputs(
        self,
        hidden_host: torch.Tensor,
        sequence_meta: H3SequenceMeta,
    ) -> None:
        self.projected_attention._validate_hidden(hidden_host)
        if hidden_host.shape[1] != self.hidden_features:
            raise ValueError("hidden_host feature size does not match the H3 runner")
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
        ops_by_device: Mapping[torch.device | str, H3BlockOps],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: MultiGpuH3DiTStats | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_host, sequence_meta)
        ops = self._normalize_ops(ops_by_device)
        stats = MultiGpuH3DiTStats() if stats is None else stats
        stats.per_device_estimated_workspace_bytes = dict(self.per_device_estimated_workspace_bytes)
        started = time.perf_counter()

        producer_ops = ops[str(self.primary_device)]
        with producer_ops.qkv_context():
            q_cpu, k_cpu, v_cpu = self.projected_attention.project_qkv_to_host(
                hidden_host,
                producer_ops.project_qkv,
                stats=stats.projection,
            )
        qkv_host_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (q_cpu, k_cpu, v_cpu)
        )
        stats.qkv_host_bytes_peak = max(stats.qkv_host_bytes_peak, qkv_host_bytes)
        raw_attention_bytes = q_cpu.numel() * q_cpu.element_size()
        stats.projection.raw_attention_roundtrip_bytes_avoided += 2 * raw_attention_bytes

        for schedule in self.attention_plan.schedules:
            device = str(schedule.device)
            device_stats = stats.per_device.setdefault(device, H3DiTStats())
            device_stats.backend = "triton"
            device_stats.estimated_workspace_bytes = self.per_device_estimated_workspace_bytes[
                device
            ]
            reset_kwargs = {}
            if self.schedule_mode == "static":
                reset_kwargs = {
                    "range_start": schedule.q_range_start,
                    "range_stop": schedule.q_range_stop,
                }
            self.consumers[device].reset(
                hidden_host=hidden_host,
                ops=ops[device],
                stats=device_stats,
                **reset_kwargs,
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
            q_tokens = (
                schedule.q_tokens
                if self.schedule_mode == "static"
                else stats.attention.dynamic_per_device[device].q_tokens
            )
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
        block_ops: Iterable[Mapping[torch.device | str, H3BlockOps]],
        *,
        softmax_scale: float | None = None,
        causal: bool = False,
        stats: MultiGpuH3DiTStats | None = None,
    ) -> torch.Tensor:
        stats = MultiGpuH3DiTStats() if stats is None else stats
        for ops_by_device in block_ops:
            self.run_block_(
                hidden_host,
                sequence_meta,
                ops_by_device,
                softmax_scale=softmax_scale,
                causal=causal,
                stats=stats,
            )
        return hidden_host


__all__ = ["MultiGpuH3DiTRunner"]
