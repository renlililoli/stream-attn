import threading
import time
from itertools import pairwise

import pytest
import torch

from seqattn_core import (
    DynamicQController,
    DynamicQueryCursor,
    DynamicScheduleConfig,
    H3BlockOps,
    H3DiTStats,
    MultiGpuAttentionStats,
    MultiGpuDeviceSpec,
    MultiGpuStreamingAttentionRunner,
    QueryTaskMeasurement,
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    build_multi_gpu_plan,
    build_plan,
)
from seqattn_core.dit.consumer import H3DeviceOutputConsumer
from seqattn_core.dit.workspace import H3BlockWorkspace
from seqattn_core.kernels import triton_is_available
from seqattn_core.reference import streaming_attention_reference
from seqattn_core.streaming.tasks import QueryTask


def _dynamic_spec(
    device: str,
    *,
    initial_q: int = 64,
    capacity_q: int = 128,
    compute_tflops: float = 100.0,
    h2d_gbps: float = 40.0,
) -> MultiGpuDeviceSpec:
    return MultiGpuDeviceSpec(
        device=device,
        config=StreamingAttentionConfig(
            q_chunk_tokens=initial_q,
            kv_chunk_tokens=64,
            block_m=16,
            block_n=16,
            backend="triton",
            pin_output=False,
            require_pinned=False,
        ),
        compute_tflops=compute_tflops,
        h2d_gbps=h2d_gbps,
        q_capacity_tokens=capacity_q,
    )


def test_dynamic_cursor_covers_packed_queries_once_with_causal_metadata():
    cursor = DynamicQueryCursor(
        [0, 19, 19, 47],
        [0, 23, 23, 57],
        active_workers=2,
    )
    tasks = []
    while True:
        task = cursor.claim(0, 16)
        if task is None:
            break
        tasks.append(task)
    cursor.retire(0)

    ordered = sorted(tasks, key=lambda task: task.q_start)
    assert ordered[0].q_start == 0
    assert ordered[-1].q_stop == 47
    assert all(left.q_stop == right.q_start for left, right in pairwise(ordered))
    assert all(task.q_stop <= (19 if task.segment_id == 0 else 47) for task in ordered)
    assert all(task.k_tokens in {23, 34} for task in ordered)
    assert all(task.causal_shift in {4, 6} for task in ordered)
    assert all(
        task.q_local_offset == task.q_start - (0 if task.segment_id == 0 else 19)
        for task in ordered
    )


def test_dynamic_cursor_concurrent_claims_have_no_gaps_or_overlap():
    cursor = DynamicQueryCursor(
        [0, 71, 128, 128, 233],
        [0, 80, 144, 144, 260],
        active_workers=4,
    )
    tasks = []
    tasks_lock = threading.Lock()

    def claim_all(device_id: int) -> None:
        try:
            while True:
                task = cursor.claim(device_id, 17 + 3 * device_id)
                if task is None:
                    return
                with tasks_lock:
                    tasks.append(task)
        finally:
            cursor.retire(device_id)

    threads = [threading.Thread(target=claim_all, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = sorted(tasks, key=lambda task: task.q_start)
    assert ordered[0].q_start == 0
    assert ordered[-1].q_stop == 233
    assert all(left.q_stop == right.q_start for left, right in pairwise(ordered))
    assert len({task.claim_order for task in tasks}) == len(tasks)


def test_dynamic_cursor_does_not_fragment_the_last_requested_q_block():
    cursor = DynamicQueryCursor([0, 64], [0, 64], active_workers=2)
    tasks = []
    while True:
        task = cursor.claim(0, 16)
        if task is None:
            break
        tasks.append(task)

    assert [task.q_tokens for task in tasks] == [16, 16, 16, 16]


def _controller(*, compute_tflops: float, h2d_gbps: float) -> DynamicQController:
    return DynamicQController(
        initial_q_tokens=64,
        q_min_tokens=32,
        q_capacity_tokens=4096,
        block_m=16,
        compute_tflops=compute_tflops,
        h2d_gbps=h2d_gbps,
        d2h_gbps=40,
        q_heads=8,
        kv_heads=8,
        element_size=2,
        config=DynamicScheduleConfig(),
    )


def test_dynamic_controller_grows_for_low_bandwidth_and_shrinks_for_low_compute():
    low_bandwidth = _controller(compute_tflops=100, h2d_gbps=10)
    high_bandwidth = _controller(compute_tflops=100, h2d_gbps=80)
    low_compute = _controller(compute_tflops=10, h2d_gbps=10)
    empty = QueryTaskMeasurement()

    for _ in range(20):
        low_bandwidth.observe(empty, update_compute=False)
        high_bandwidth.observe(empty, update_compute=False)
        low_compute.observe(empty, update_compute=False)

    assert low_bandwidth.q_current > high_bandwidth.q_current
    assert low_compute.q_current < low_bandwidth.q_current
    assert low_bandwidth.q_current <= 4096
    assert low_bandwidth.q_current % 16 == 0


def test_dynamic_controller_respects_threshold_step_limit_and_truncated_samples():
    controller = _controller(compute_tflops=100, h2d_gbps=10)
    previous = controller.q_current
    _, updated = controller.observe(QueryTaskMeasurement(), update_compute=False)
    assert updated <= previous * 1.25 + 16

    compute_before = controller.effective_tflops_ema
    controller.observe(
        QueryTaskMeasurement(
            h2d_seconds=0.01,
            h2d_bytes=50_000_000,
            attention_seconds=1.0,
            attention_flops=1,
            elapsed_seconds=1.0,
        ),
        update_compute=False,
    )
    assert controller.effective_tflops_ema == compute_before
    assert controller.h2d_gbps_ema != 10


def test_dynamic_controller_can_grow_from_5760_with_preallocated_headroom():
    controller = DynamicQController(
        initial_q_tokens=5760,
        q_min_tokens=128,
        q_capacity_tokens=11520,
        block_m=128,
        compute_tflops=400,
        h2d_gbps=10,
        d2h_gbps=40,
        q_heads=56,
        kv_heads=56,
        element_size=2,
        config=DynamicScheduleConfig(),
    )

    for _ in range(20):
        controller.observe(QueryTaskMeasurement(), update_compute=False)

    assert 5760 < controller.q_current <= 11520


def test_dynamic_controller_supports_a_sequence_shorter_than_block_m():
    controller = DynamicQController(
        initial_q_tokens=8,
        q_min_tokens=8,
        q_capacity_tokens=8,
        block_m=16,
        compute_tflops=10,
        h2d_gbps=10,
        d2h_gbps=10,
        q_heads=2,
        kv_heads=2,
        element_size=2,
        config=DynamicScheduleConfig(),
    )

    assert controller.q_current == 8
    assert controller.observe(QueryTaskMeasurement(), update_compute=False) == (8, 8)


class _FakeDynamicRunner:
    def __init__(self, plan, *, delay_seconds: float, fail_first: bool = False) -> None:
        self.plan = plan
        self.delay_seconds = delay_seconds
        self.fail_first = fail_first
        self.tasks = []

    def run_query_tasks(
        self,
        q_cpu,
        k_cpu,
        v_cpu,
        query_tasks,
        *,
        out,
        stats,
        task_measurement,
        **kwargs,
    ):
        del k_cpu, v_cpu, kwargs
        task = query_tasks[0]
        self.tasks.append(task)
        if self.fail_first and len(self.tasks) == 1:
            raise RuntimeError("injected dynamic worker failure")
        time.sleep(self.delay_seconds)
        out[task.q_start : task.q_stop].copy_(q_cpu[task.q_start : task.q_stop])
        task_measurement.elapsed_seconds = self.delay_seconds
        task_measurement.attention_seconds = self.delay_seconds
        task_measurement.h2d_seconds = self.delay_seconds
        task_measurement.h2d_bytes = 1_000_000
        task_measurement.attention_flops = 1_000_000_000
        stats.q_chunks += 1
        return out


def _fake_dynamic_plan(tokens: int):
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            _dynamic_spec("cuda:0"),
            _dynamic_spec("cuda:1"),
        ],
        schedule_mode="dynamic",
        dynamic_config=DynamicScheduleConfig(enable_task_trace=True),
    )
    return cu, plan


def test_static_plan_can_preallocate_capacity_for_runner_dynamic_opt_in():
    tokens = 256
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=2,
        kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[_dynamic_spec("cuda:0"), _dynamic_spec("cuda:1")],
    )

    assert all(schedule.attention_plan.q_chunk_tokens == 128 for schedule in plan.schedules)
    assert all(task.q_tokens <= 64 for schedule in plan.schedules for task in schedule.query_tasks)
    overrides = {
        schedule.device: _FakeDynamicRunner(schedule.attention_plan, delay_seconds=0.001)
        for schedule in plan.schedules
    }
    runner = MultiGpuStreamingAttentionRunner(
        plan,
        runner_overrides=overrides,
        schedule_mode="dynamic",
    )
    q = torch.randn(tokens, 2, 16)
    try:
        actual = runner(q, q, q, cu, cu, out=torch.empty_like(q))
    finally:
        runner.close()
    torch.testing.assert_close(actual, q)


def test_dynamic_plan_rejects_q_min_above_aligned_capacity():
    tokens = 8192
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    devices = [
        MultiGpuDeviceSpec(
            device=f"cuda:{index}",
            config=StreamingAttentionConfig(
                q_chunk_tokens=5760,
                kv_chunk_tokens=64,
                block_m=128,
                block_n=16,
                backend="triton",
            ),
            compute_tflops=100,
            h2d_gbps=40,
            q_min_tokens=5999,
            q_capacity_tokens=6000,
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="aligned Q capacity"):
        build_multi_gpu_plan(
            q_heads=2,
            kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            max_q_tokens=tokens,
            max_kv_tokens=tokens,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            devices=devices,
            schedule_mode="dynamic",
        )


def test_dynamic_runner_assigns_more_tokens_to_the_faster_worker():
    tokens = 1024
    cu, plan = _fake_dynamic_plan(tokens)
    fast = _FakeDynamicRunner(plan.schedules[0].attention_plan, delay_seconds=0.001)
    slow = _FakeDynamicRunner(plan.schedules[1].attention_plan, delay_seconds=0.01)
    runner = MultiGpuStreamingAttentionRunner(
        plan,
        runner_overrides={"cuda:0": fast, "cuda:1": slow},
    )
    q = torch.randn(tokens, 2, 16)
    stats = MultiGpuAttentionStats()
    try:
        actual = runner(q, q, q, cu, cu, out=torch.empty_like(q), stats=stats)
    finally:
        runner.close()

    torch.testing.assert_close(actual, q)
    assert stats.dynamic_per_device["cuda:0"].q_tokens > stats.dynamic_per_device["cuda:1"].q_tokens
    assert sum(item.q_tokens for item in stats.dynamic_per_device.values()) == tokens
    assert [item.claim_order for item in stats.task_trace] == list(range(len(stats.task_trace)))


def test_dynamic_runner_cancels_new_claims_after_worker_failure():
    tokens = 2048
    cu, plan = _fake_dynamic_plan(tokens)
    failing = _FakeDynamicRunner(
        plan.schedules[0].attention_plan,
        delay_seconds=0.001,
        fail_first=True,
    )
    peer = _FakeDynamicRunner(plan.schedules[1].attention_plan, delay_seconds=0.02)
    runner = MultiGpuStreamingAttentionRunner(
        plan,
        runner_overrides={"cuda:0": failing, "cuda:1": peer},
    )
    q = torch.randn(tokens, 2, 16)
    try:
        with pytest.raises(RuntimeError, match="injected dynamic worker failure"):
            runner(q, q, q, cu, cu, out=torch.empty_like(q))
    finally:
        runner.close()

    assert len(failing.tasks) == 1
    assert len(peer.tasks) <= 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_is_available(),
    reason="requires CUDA and Triton",
)
def test_single_dynamic_query_task_reports_cuda_timing_and_matches_reference():
    torch.manual_seed(307)
    dtype = torch.bfloat16
    q = torch.randn(33, 4, 32, dtype=dtype, pin_memory=True)
    k = torch.randn(41, 2, 32, dtype=dtype, pin_memory=True)
    v = torch.randn_like(k).pin_memory()
    config = StreamingAttentionConfig(
        q_chunk_tokens=32,
        kv_chunk_tokens=16,
        block_m=16,
        block_n=16,
        backend="triton",
    )
    plan = build_plan(
        q_heads=4,
        kv_heads=2,
        head_dim=32,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        config=config,
    )
    task = QueryTask(
        q_start=7,
        q_stop=23,
        k_start=0,
        k_stop=41,
        q_local_offset=7,
        causal_shift=8,
        segment_id=0,
    )
    out = torch.empty_like(q, pin_memory=True)
    measurement = QueryTaskMeasurement()
    runner = StreamingAttentionRunner(plan, config)
    runner.run_query_tasks(
        q,
        k,
        v,
        (task,),
        causal=True,
        out=out,
        task_measurement=measurement,
    )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        torch.tensor([0, 33], dtype=torch.int32),
        torch.tensor([0, 41], dtype=torch.int32),
        q_chunk_tokens=16,
        kv_chunk_tokens=16,
        device="cuda:0",
        causal=True,
    )

    torch.testing.assert_close(out[7:23], expected[7:23], atol=6e-2, rtol=8e-3)
    assert measurement.elapsed_seconds > 0
    assert measurement.h2d_seconds > 0
    assert measurement.attention_seconds > 0
    assert measurement.d2h_seconds > 0
    assert measurement.h2d_bytes > 0
    assert measurement.d2h_bytes == out[7:23].numel() * out.element_size()
    assert runner._workspace is not None
    assert runner._workspace.task_timing is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_h3_consumer_runs_ffn_once_per_dynamic_q_task():
    dtype = torch.float16
    hidden_features = 16
    hidden = torch.zeros(32, hidden_features, dtype=dtype, pin_memory=True)
    workspace = H3BlockWorkspace(
        hidden_features=hidden_features,
        mlp_chunk_tokens=4,
        dtype=dtype,
        device=torch.device("cuda:0"),
        final_output_chunk_tokens=8,
    )
    stats = H3DiTStats()
    ops = H3BlockOps(
        project_qkv=lambda *args: args,
        attention_epilogue=lambda attention, start, stop: attention,
        mlp=lambda tile, start, stop: tile.add(1),
    )
    consumer = H3DeviceOutputConsumer(workspace)
    consumer.reset(hidden_host=hidden, ops=ops, stats=stats)
    tasks = (
        QueryTask(0, 7, 0, 8, 0, 1, segment_id=0),
        QueryTask(17, 24, 8, 16, 1, 0, segment_id=1),
    )
    values = (2.0, 5.0)
    with torch.cuda.device("cuda:0"), torch.cuda.stream(torch.cuda.current_stream("cuda:0")):
        for task, value in zip(tasks, values):
            consumer.begin_task(task)
            attention = torch.full(
                (task.q_tokens, hidden_features),
                value,
                dtype=dtype,
                device="cuda:0",
            )
            consumer(attention, task.q_start, task.q_stop)
            consumer.finish_task().synchronize()

    assert torch.count_nonzero(hidden[7:17]) == 0
    assert torch.count_nonzero(hidden[24:]) == 0
    torch.testing.assert_close(hidden[0:7], torch.full_like(hidden[0:7], 3.0))
    torch.testing.assert_close(hidden[17:24], torch.full_like(hidden[17:24], 6.0))
    assert stats.mlp_cross_q_boundaries == 0
    assert stats.mlp_chunks == 2
    assert consumer.task_d2h_bytes() == 7 * hidden_features * hidden.element_size()
