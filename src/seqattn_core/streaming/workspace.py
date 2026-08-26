from __future__ import annotations

import torch

from ..planner import AttentionPlan


class TaskTimingEvents:
    def __init__(self) -> None:
        self.task_start = torch.cuda.Event(enable_timing=True)
        self.h2d_start = torch.cuda.Event(enable_timing=True)
        self.h2d_end = torch.cuda.Event(enable_timing=True)
        self.attention_start = torch.cuda.Event(enable_timing=True)
        self.attention_end = torch.cuda.Event(enable_timing=True)
        self.consumer_start = torch.cuda.Event(enable_timing=True)
        self.consumer_end = torch.cuda.Event(enable_timing=True)
        self.d2h_start = torch.cuda.Event(enable_timing=True)
        self.d2h_end = torch.cuda.Event(enable_timing=True)
        self.task_done = torch.cuda.Event(enable_timing=True)


class CudaWorkspace:
    def __init__(self, plan: AttentionPlan) -> None:
        device = plan.device
        self.q = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=plan.dtype,
            device=device,
        )
        self.k = [
            torch.empty(
                (plan.kv_chunk_tokens, plan.kv_heads, plan.head_dim),
                dtype=plan.dtype,
                device=device,
            )
            for _ in range(plan.num_kv_buffers)
        ]
        self.v = [torch.empty_like(tensor) for tensor in self.k]
        self.running_max = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads), dtype=torch.float32, device=device
        )
        self.running_sum = torch.empty_like(self.running_max)
        self.accumulator = torch.empty(
            (plan.q_chunk_tokens, plan.q_heads, plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.output = (
            [torch.empty_like(self.q) for _ in range(plan.num_output_buffers)]
            if plan.output_mode == "host"
            else []
        )
        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.q_ready = torch.cuda.Event()
        self.q_free = torch.cuda.Event()
        self.q_has_pending_compute = False
        self.kv_ready = [torch.cuda.Event() for _ in self.k]
        self.kv_free = [torch.cuda.Event() for _ in self.k]
        self.kv_has_pending_compute = [False for _ in self.k]
        self.output_ready = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_free = [torch.cuda.Event() for _ in range(plan.num_output_buffers)]
        self.output_has_pending_copy = [False for _ in range(plan.num_output_buffers)]
        self.pipeline_start = torch.cuda.Event(enable_timing=True)
        self.pipeline_end = torch.cuda.Event(enable_timing=True)
        self.task_timing: TaskTimingEvents | None = None

    def get_task_timing(self) -> TaskTimingEvents:
        if self.task_timing is None:
            self.task_timing = TaskTimingEvents()
        return self.task_timing


__all__ = ["CudaWorkspace", "TaskTimingEvents"]
