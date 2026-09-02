import copy
import math
from itertools import pairwise

import pytest
import torch
from seqattn_core import (
    ProjectedAttentionRunner,
    ProjectedAttentionStats,
    ProjectionPipelineConfig,
    StreamingAttentionConfig,
    build_attention_plan,
    build_query_tasks,
)
from seqattn_core._plugin_api import H3BlockOps, H3MaterializedProjection, H3SequenceMeta
from seqattn_core.kernels import triton_is_available
from seqattn_core.reference import streaming_attention_reference

from seqattn_multigpu import (
    DynamicScheduleConfig,
    MultiGpuAttentionStats,
    MultiGpuDeviceSpec,
    MultiGpuH3DiTStats,
    MultiGpuH3MaterializedRunner,
    MultiGpuQKVProjectionRunner,
    MultiGpuStreamingAttentionRunner,
    build_multi_gpu_plan,
)


def device_spec(
    device: str,
    *,
    q_chunk: int,
    kv_chunk: int,
    compute_tflops: float,
    h2d_gbps: float,
    output_mode: str = "host",
) -> MultiGpuDeviceSpec:
    return MultiGpuDeviceSpec(
        device=device,
        config=StreamingAttentionConfig(
            q_chunk_tokens=q_chunk,
            kv_chunk_tokens=kv_chunk,
            block_m=16,
            block_n=16,
            backend="triton",
            output_mode=output_mode,
        ),
        compute_tflops=compute_tflops,
        h2d_gbps=h2d_gbps,
    )


def test_query_tasks_preserve_packed_offsets_for_a_subrange():
    tasks = build_query_tasks(
        [0, 7, 7, 20],
        [0, 9, 9, 25],
        q_chunk_tokens=5,
        range_start=4,
        range_stop=16,
    )

    assert [
        (
            task.q_start,
            task.q_stop,
            task.k_start,
            task.k_stop,
            task.q_local_offset,
            task.causal_shift,
        )
        for task in tasks
    ] == [
        (4, 7, 0, 9, 4, 2),
        (7, 12, 9, 25, 0, 3),
        (12, 16, 9, 25, 5, 3),
    ]


def test_static_plan_assigns_more_queries_to_the_faster_device():
    tokens = 16_384
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            device_spec(
                "cuda:0",
                q_chunk=4096,
                kv_chunk=1024,
                compute_tflops=200,
                h2d_gbps=50,
            ),
            device_spec(
                "cuda:1",
                q_chunk=2048,
                kv_chunk=2048,
                compute_tflops=100,
                h2d_gbps=25,
            ),
        ],
    )

    first, second = plan.schedules
    assert first.q_range_start == 0
    assert first.q_range_stop == second.q_range_start
    assert second.q_range_stop == tokens
    assert first.q_tokens > second.q_tokens
    assert max(task.q_tokens for task in first.query_tasks) <= 4096
    assert max(task.q_tokens for task in second.query_tasks) <= 2048
    assert first.attention_plan.kv_chunk_tokens == 1024
    assert second.attention_plan.kv_chunk_tokens == 2048
    assert (
        max(first.estimated_seconds, second.estimated_seconds)
        / min(first.estimated_seconds, second.estimated_seconds)
        < 1.3
    )


def test_static_plan_accounts_for_device_bandwidth():
    tokens = 8192
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            device_spec(
                "cuda:0",
                q_chunk=1024,
                kv_chunk=1024,
                compute_tflops=1000,
                h2d_gbps=60,
            ),
            device_spec(
                "cuda:1",
                q_chunk=1024,
                kv_chunk=1024,
                compute_tflops=1000,
                h2d_gbps=20,
            ),
        ],
    )

    assert plan.schedules[0].q_tokens > plan.schedules[1].q_tokens


def test_static_plan_tasks_cover_packed_queries_exactly_once():
    cu_q = torch.tensor([0, 1536, 1536, 4096], dtype=torch.int32)
    cu_k = torch.tensor([0, 2048, 2048, 5120], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=4,
        kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        max_q_tokens=4096,
        max_kv_tokens=5120,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        devices=[
            device_spec(
                "cuda:0",
                q_chunk=768,
                kv_chunk=512,
                compute_tflops=150,
                h2d_gbps=40,
            ),
            device_spec(
                "cuda:1",
                q_chunk=512,
                kv_chunk=1024,
                compute_tflops=120,
                h2d_gbps=35,
            ),
        ],
    )

    tasks = [task for schedule in plan.schedules for task in schedule.query_tasks]
    tasks.sort(key=lambda task: task.q_start)
    assert tasks[0].q_start == 0
    assert tasks[-1].q_stop == 4096
    assert all(left.q_stop == right.q_start for left, right in pairwise(tasks))
    assert all(task.k_tokens in {2048, 3072} for task in tasks)
    assert all(task.q_local_offset >= 0 for task in tasks)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("schedule_mode", ["static", "dynamic"])
def test_two_gpu_runner_matches_reference(causal, schedule_mode):
    torch.manual_seed(211)
    dtype = torch.bfloat16
    q = torch.randn(257, 4, 32, dtype=dtype, pin_memory=True)
    k = torch.randn(293, 2, 32, dtype=dtype, pin_memory=True)
    v = torch.randn_like(k).pin_memory()
    cu_q = torch.tensor([0, 101, 257], dtype=torch.int32)
    cu_k = torch.tensor([0, 119, 293], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=4,
        kv_heads=2,
        head_dim=32,
        dtype=dtype,
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        devices=[
            device_spec(
                "cuda:0",
                q_chunk=48,
                kv_chunk=64,
                compute_tflops=200,
                h2d_gbps=50,
            ),
            device_spec(
                "cuda:1",
                q_chunk=80,
                kv_chunk=48,
                compute_tflops=140,
                h2d_gbps=35,
            ),
        ],
        schedule_mode=schedule_mode,
        dynamic_config=DynamicScheduleConfig(enable_task_trace=True),
    )
    runner = MultiGpuStreamingAttentionRunner(plan)
    stats = MultiGpuAttentionStats()
    try:
        actual = runner(q, k, v, cu_q, cu_k, causal=causal, stats=stats)
    finally:
        runner.close()
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=47,
        kv_chunk_tokens=53,
        device="cuda:0",
        causal=causal,
    )

    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=8e-3)
    assert set(stats.per_device) == {"cuda:0", "cuda:1"}
    if schedule_mode == "static":
        assert sum(device_stats.q_chunks for device_stats in stats.per_device.values()) == sum(
            len(schedule.query_tasks) for schedule in plan.schedules
        )
    else:
        assert sum(item.q_tokens for item in stats.dynamic_per_device.values()) == q.shape[0]
        assert sum(item.task_count for item in stats.dynamic_per_device.values()) == len(
            stats.task_trace
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
def test_two_gpu_static_device_consumers_match_reference():
    torch.manual_seed(223)
    dtype = torch.bfloat16
    tokens = 193
    q = torch.randn(tokens, 4, 32, dtype=dtype, pin_memory=True)
    k = torch.randn_like(q).pin_memory()
    v = torch.randn_like(q).pin_memory()
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    plan = build_multi_gpu_plan(
        q_heads=4,
        kv_heads=4,
        head_dim=32,
        dtype=dtype,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            device_spec(
                "cuda:0",
                q_chunk=48,
                kv_chunk=64,
                compute_tflops=180,
                h2d_gbps=45,
                output_mode="device_consumer",
            ),
            device_spec(
                "cuda:1",
                q_chunk=64,
                kv_chunk=48,
                compute_tflops=150,
                h2d_gbps=40,
                output_mode="device_consumer",
            ),
        ],
    )

    class RangeConsumer:
        def __init__(self, schedule):
            self.device = schedule.device
            self.range_start = schedule.q_range_start
            self.range_stop = schedule.q_range_stop
            self.output = torch.empty(
                (schedule.q_tokens, q.shape[1] * q.shape[2]),
                dtype=q.dtype,
                device=schedule.device,
            )

        def __call__(self, attention, start, stop):
            self.output[start - self.range_start : stop - self.range_start].copy_(attention)

        def finish(self):
            pass

        def synchronize(self):
            torch.cuda.synchronize(self.device)

    consumers = {schedule.device: RangeConsumer(schedule) for schedule in plan.schedules}
    runner = MultiGpuStreamingAttentionRunner(plan)
    try:
        runner.run_with_device_consumers(
            q,
            k,
            v,
            cu,
            cu,
            output_consumers=consumers,
        )
    finally:
        runner.close()
    actual = torch.empty_like(q)
    for schedule in plan.schedules:
        consumer = consumers[schedule.device]
        actual[schedule.q_range_start : schedule.q_range_stop].copy_(
            consumer.output.reshape(schedule.q_tokens, *q.shape[1:]).cpu()
        )
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu,
        cu,
        q_chunk_tokens=47,
        kv_chunk_tokens=53,
        device="cuda:0",
    )

    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=8e-3)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
def test_two_gpu_qkv_projection_uses_default_4096_token_blocks():
    torch.manual_seed(225)
    dtype = torch.bfloat16
    tokens = 8201
    hidden_features = 16
    heads = 2
    head_dim = 16
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    base_qkv = torch.nn.Linear(hidden_features, heads * head_dim * 3, bias=False)
    qkv_by_device = {
        device: copy.deepcopy(base_qkv).to(device=device, dtype=dtype)
        for device in (torch.device("cuda:0"), torch.device("cuda:1"))
    }
    primary_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=64,
        kv_chunk_tokens=64,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    primary_plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=primary_config,
    )
    projected = ProjectedAttentionRunner(primary_plan)
    multi_plan = build_multi_gpu_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            MultiGpuDeviceSpec(
                device="cuda:0",
                config=primary_config,
                compute_tflops=180,
                h2d_gbps=45,
            ),
            device_spec(
                "cuda:1",
                q_chunk=64,
                kv_chunk=64,
                compute_tflops=160,
                h2d_gbps=40,
                output_mode="device_consumer",
            ),
        ],
    )
    runner = MultiGpuQKVProjectionRunner(
        projected,
        multi_plan,
        hidden_features=hidden_features,
    )
    ranges = {str(device): [] for device in qkv_by_device}

    def make_projection(device):
        def project_qkv(tile, start, stop):
            ranges[str(device)].append((start, stop))
            qkv = qkv_by_device[device](tile).view(-1, 3, heads, head_dim)
            return tuple(qkv[:, index].contiguous() for index in range(3))

        return H3MaterializedProjection(project_qkv)

    stats = ProjectedAttentionStats()
    per_device_stats = {device: ProjectedAttentionStats() for device in ranges}
    try:
        q_cpu, k_cpu, v_cpu = runner.run(
            hidden,
            {str(device): make_projection(device) for device in qkv_by_device},
            stats=stats,
            per_device_stats=per_device_stats,
        )
    finally:
        runner.close()

    expected_qkv = qkv_by_device[torch.device("cuda:0")](hidden.to("cuda:0")).view(
        tokens, 3, heads, head_dim
    )
    torch.testing.assert_close(q_cpu, expected_qkv[:, 0].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(k_cpu, expected_qkv[:, 1].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(v_cpu, expected_qkv[:, 2].cpu(), atol=0, rtol=0)
    assert stats.projection_chunks == 3
    assert stats.projection_tokens == tokens
    assert sorted(
        stop - start for device_ranges in ranges.values() for start, stop in device_ranges
    ) == [
        9,
        4096,
        4096,
    ]
    assert all(item.projection_tokens > 0 for item in per_device_stats.values())


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
def test_two_gpu_h3_runner_dynamically_projects_and_matches_full_gpu_block():
    torch.manual_seed(227)
    dtype = torch.bfloat16
    tokens = 97
    hidden_features = 48
    heads = 4
    head_dim = 16
    inner = heads * head_dim
    mlp_features = 80
    cu = torch.tensor([0, 37, 97], dtype=torch.int32)
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = hidden.clone()

    base_modules = (
        torch.nn.Linear(hidden_features, inner * 3, bias=False),
        torch.nn.Linear(inner, hidden_features, bias=False),
        torch.nn.Linear(hidden_features, mlp_features * 2, bias=False),
        torch.nn.Linear(mlp_features, hidden_features, bias=False),
    )
    modules = {
        device: tuple(
            copy.deepcopy(module).to(device=device, dtype=dtype) for module in base_modules
        )
        for device in (torch.device("cuda:0"), torch.device("cuda:1"))
    }

    primary_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=29,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    primary_plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=primary_config,
    )
    projected = ProjectedAttentionRunner(
        primary_plan,
        ProjectionPipelineConfig(projection_tile_tokens=31),
    )
    multi_plan = build_multi_gpu_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            MultiGpuDeviceSpec(
                device="cuda:0",
                config=primary_config,
                compute_tflops=180,
                h2d_gbps=45,
            ),
            device_spec(
                "cuda:1",
                q_chunk=48,
                kv_chunk=31,
                compute_tflops=140,
                h2d_gbps=35,
                output_mode="device_consumer",
            ),
        ],
    )
    runner = MultiGpuH3MaterializedRunner(
        projected,
        multi_plan,
        hidden_features=hidden_features,
        projection_tile_tokens=31,
    )

    projection_ranges = {device: [] for device in modules}

    def make_block(device):
        qkv_linear, out_linear, fc1, fc2 = modules[device]

        def project_qkv(tile, start, stop):
            projection_ranges[device].append((start, stop))
            qkv = qkv_linear(tile).view(-1, 3, heads, head_dim)
            return tuple(qkv[:, index].contiguous() for index in range(3))

        def attention_epilogue(attention, residual_host, start, stop):
            residual = residual_host[start:stop].to(device, non_blocking=True)
            return out_linear(attention).add_(residual)

        def mlp(post_attention, start, stop):
            del start, stop
            gate, up = fc1(post_attention).chunk(2, dim=-1)
            return post_attention.add_(fc2(torch.nn.functional.silu(gate).mul_(up)))

        return H3MaterializedProjection(project_qkv), H3BlockOps(attention_epilogue, mlp)

    blocks = {device: make_block(device) for device in modules}

    stats = MultiGpuH3DiTStats()
    try:
        runner.run_block_(
            hidden,
            H3SequenceMeta(cu),
            {device: block[0] for device, block in blocks.items()},
            {device: block[1] for device, block in blocks.items()},
            softmax_scale=head_dim**-0.5,
            stats=stats,
        )
    finally:
        runner.close()

    qkv_linear, out_linear, fc1, fc2 = modules[torch.device("cuda:0")]
    hidden_gpu = original.to("cuda:0")
    qkv = qkv_linear(hidden_gpu).view(tokens, 3, heads, head_dim)
    q, k, v = (qkv[:, index] for index in range(3))
    expected_attention = torch.empty_like(q)
    for start, stop in pairwise(cu.tolist()):
        tile = torch.nn.functional.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1).unsqueeze(0),
            k[start:stop].transpose(0, 1).unsqueeze(0),
            v[start:stop].transpose(0, 1).unsqueeze(0),
            scale=head_dim**-0.5,
        )
        expected_attention[start:stop].copy_(tile.squeeze(0).transpose(0, 1))
    expected = out_linear(expected_attention.reshape(tokens, inner)).add_(hidden_gpu)
    gate, up = fc1(expected).chunk(2, dim=-1)
    expected = expected + fc2(torch.nn.functional.silu(gate) * up)

    torch.testing.assert_close(hidden, expected.cpu(), atol=7e-2, rtol=1e-2)
    assert stats.blocks == 1
    assert stats.final_hidden_d2h_bytes == hidden.numel() * hidden.element_size()
    assert set(stats.per_device) == {"cuda:0", "cuda:1"}
    assert all(device_stats.blocks == 1 for device_stats in stats.per_device.values())
    assert all(
        device_stats.final_hidden_d2h_bytes > 0 for device_stats in stats.per_device.values()
    )
    assert all(device_stats.d2h_bytes == 0 for device_stats in stats.attention.per_device.values())
    assert multi_plan.schedule_mode == "static"
    assert runner.attention.schedule_mode == "dynamic"
    assert stats.projection.projection_chunks == math.ceil(tokens / 31)
    assert stats.projection.projection_tokens == tokens
    assert all(projection_ranges[device] for device in modules)
    assert (
        sum(device_stats.projection.projection_tokens for device_stats in stats.per_device.values())
        == tokens
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not triton_is_available(),
    reason="requires two CUDA devices and Triton",
)
def test_two_gpu_h3_projection_failure_stops_before_attention():
    dtype = torch.bfloat16
    tokens = 65
    hidden_features = 16
    heads = 2
    head_dim = 16
    cu = torch.tensor([0, tokens], dtype=torch.int32)
    hidden = torch.randn(tokens, hidden_features, dtype=dtype, pin_memory=True)
    original = hidden.clone()
    primary_config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=32,
        kv_chunk_tokens=32,
        block_m=16,
        block_n=16,
        output_mode="device_consumer",
    )
    primary_plan = build_attention_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cuda:0",
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        config=primary_config,
    )
    projected = ProjectedAttentionRunner(primary_plan)
    multi_plan = build_multi_gpu_plan(
        q_heads=heads,
        kv_heads=heads,
        head_dim=head_dim,
        dtype=dtype,
        max_q_tokens=tokens,
        max_kv_tokens=tokens,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        devices=[
            MultiGpuDeviceSpec(
                device="cuda:0",
                config=primary_config,
                compute_tflops=180,
                h2d_gbps=45,
            ),
            device_spec(
                "cuda:1",
                q_chunk=32,
                kv_chunk=32,
                compute_tflops=160,
                h2d_gbps=40,
                output_mode="device_consumer",
            ),
        ],
    )
    runner = MultiGpuH3MaterializedRunner(
        projected,
        multi_plan,
        hidden_features=hidden_features,
        projection_tile_tokens=16,
    )

    def fail_projection(*_args):
        raise RuntimeError("injected QKV projection failure")

    def unexpected_consumer(*_args):
        raise AssertionError("attention consumer must not run after projection failure")

    projection = H3MaterializedProjection(fail_projection)
    ops = H3BlockOps(unexpected_consumer, unexpected_consumer)
    stats = MultiGpuH3DiTStats()
    try:
        with pytest.raises(RuntimeError, match="injected QKV projection failure"):
            runner.run_block_(
                hidden,
                H3SequenceMeta(cu),
                {"cuda:0": projection, "cuda:1": projection},
                {"cuda:0": ops, "cuda:1": ops},
                stats=stats,
            )
    finally:
        runner.close()

    torch.testing.assert_close(hidden, original)
    assert stats.attention.wall_seconds == 0
