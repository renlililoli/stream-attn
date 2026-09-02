from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ...plan import AttentionPlan
from ..layout import KVLayout, PageDescriptor, TensorLayout
from ..memory_budget import HostMemoryPlan
from .types import KVStage


class HostStaging:
    def __init__(
        self,
        q_layout: TensorLayout,
        kv_layout: KVLayout,
        q_pages: Sequence[PageDescriptor],
        kv_pages: Sequence[PageDescriptor],
        *,
        queue_depth: int,
        output_buffers: int,
        pinned: bool,
        memory_plan: HostMemoryPlan,
    ) -> None:
        self.memory_plan = memory_plan
        self.registered_bytes = 0
        max_q_tokens = max((page.padded_tokens for page in q_pages), default=1)
        max_kv_tokens = max((page.padded_tokens for page in kv_pages), default=1)
        self.q = torch.empty(
            (max_q_tokens, q_layout.heads, q_layout.head_dim),
            dtype=q_layout.torch_dtype,
            pin_memory=pinned,
        )
        self.outputs = [torch.empty_like(self.q, pin_memory=pinned) for _ in range(output_buffers)]
        self.kv: list[KVStage] = []
        scale_groups = math.ceil(max_kv_tokens / kv_layout.quant_group_tokens)
        for _ in range(queue_depth):
            k = torch.empty(
                (max_kv_tokens, kv_layout.heads, kv_layout.head_dim),
                dtype=kv_layout.storage_torch_dtype,
                pin_memory=pinned,
            )
            v = torch.empty_like(k, pin_memory=pinned)
            if kv_layout.storage_dtype == "int8":
                k_scales = torch.empty(
                    (scale_groups, kv_layout.heads),
                    dtype=torch.float16,
                    pin_memory=pinned,
                )
                v_scales = torch.empty_like(k_scales, pin_memory=pinned)
            else:
                k_scales = v_scales = None
            self.kv.append(KVStage(k, v, k_scales, v_scales))
        tensors = [self.q, *self.outputs]
        for stage in self.kv:
            tensors.extend((stage.k, stage.v))
            if stage.k_scales is not None and stage.v_scales is not None:
                tensors.extend((stage.k_scales, stage.v_scales))
        self.registered_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        try:
            memory_plan.register("pinned", self.registered_bytes)
        except BaseException:
            self.q = torch.empty(0)
            self.outputs = []
            self.kv = []
            raise

    def close(self) -> None:
        if not self.registered_bytes:
            return
        self.q = torch.empty(0)
        self.outputs = []
        self.kv = []
        self.memory_plan.release("pinned", self.registered_bytes)
        self.registered_bytes = 0


class PagedCudaWorkspace:
    def __init__(self, plan: AttentionPlan, kv_layout: KVLayout) -> None:
        device = plan.device
        self.q = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device=device,
        )
        self.k = [
            torch.empty(
                (plan.kv_chunk_tokens, plan.kv_heads, plan.head_dim),
                dtype=kv_layout.storage_torch_dtype,
                device=device,
            )
            for _ in range(plan.num_kv_buffers)
        ]
        self.v = [torch.empty_like(tensor) for tensor in self.k]
        if kv_layout.storage_dtype == "int8":
            groups = math.ceil(
                (plan.kv_chunk_tokens + kv_layout.quant_group_tokens - 1)
                / kv_layout.quant_group_tokens
            )
            self.k_scales = [
                torch.empty((groups, plan.kv_heads), dtype=torch.float16, device=device)
                for _ in self.k
            ]
            self.v_scales = [torch.empty_like(tensor) for tensor in self.k_scales]
        else:
            self.k_scales = self.v_scales = None
        self.running_max = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads), dtype=torch.float32, device=device
        )
        self.running_sum = torch.empty_like(self.running_max)
        self.accumulator = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.output = [torch.empty_like(self.q) for _ in range(plan.num_output_buffers)]
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.q_ready = torch.cuda.Event()
        self.q_free = torch.cuda.Event()
        self.q_busy = False
        self.kv_ready = [torch.cuda.Event() for _ in self.k]
        self.kv_free = [torch.cuda.Event() for _ in self.k]
        self.kv_busy = [False for _ in self.k]
        self.output_ready = [torch.cuda.Event() for _ in self.output]
        self.output_free = [torch.cuda.Event() for _ in self.output]
        self.output_busy = [False for _ in self.output]
        self.stage_free: list[torch.cuda.Event] = []
        self.stage_busy: list[bool] = []
        self.output_host_ready: list[torch.cuda.Event] = []

    def bind_host_rings(self, staging: HostStaging) -> None:
        self.stage_free = [torch.cuda.Event() for _ in staging.kv]
        self.stage_busy = [False for _ in staging.kv]
        self.output_host_ready = [torch.cuda.Event() for _ in staging.outputs]


__all__ = ["HostStaging", "PagedCudaWorkspace"]
