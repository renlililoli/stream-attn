import pytest
import torch

from seqattn import (
    StreamingAttentionConfig,
    StreamingAttentionRunner,
    build_plan,
)
from seqattn.kernels import triton_is_available
from seqattn.reference import streaming_attention_reference


pytestmark = pytest.mark.skipif(
    not triton_is_available(), reason="requires CUDA and Triton"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (8, 2)])
def test_triton_matches_reference(dtype, causal, q_heads, kv_heads):
    torch.manual_seed(17)
    q = torch.randn(73, q_heads, 64, dtype=dtype, pin_memory=True)
    k = torch.randn(89, kv_heads, 64, dtype=dtype, pin_memory=True)
    v = torch.randn(89, kv_heads, 64, dtype=dtype, pin_memory=True)
    cu_q = torch.tensor([0, 31, 31, 73], dtype=torch.int32)
    cu_k = torch.tensor([0, 37, 37, 89], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=23,
        kv_chunk_tokens=19,
        block_m=32,
        block_n=32,
    )
    plan = build_plan(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=64,
        dtype=dtype,
        device="cuda",
        max_q_tokens=q.shape[0],
        max_kv_tokens=k.shape[0],
        config=config,
    )
    runner = StreamingAttentionRunner(plan, config)
    actual = runner(q, k, v, cu_q, cu_k, causal=causal)
    expected = streaming_attention_reference(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q_chunk_tokens=23,
        kv_chunk_tokens=19,
        device="cuda",
        causal=causal,
    )
    torch.testing.assert_close(actual, expected, atol=6e-2, rtol=8e-3)


def test_runner_reuse_has_bounded_allocator_growth():
    torch.manual_seed(19)
    q = torch.randn(257, 4, 64, dtype=torch.bfloat16, pin_memory=True)
    k = torch.randn_like(q)
    k = k.pin_memory()
    v = torch.randn(257, 4, 64, dtype=torch.bfloat16, pin_memory=True)
    cu = torch.tensor([0, 257], dtype=torch.int32)
    config = StreamingAttentionConfig(
        backend="triton",
        q_chunk_tokens=128,
        kv_chunk_tokens=96,
        block_m=32,
        block_n=32,
    )
    plan = build_plan(
        q_heads=4,
        kv_heads=4,
        head_dim=64,
        dtype=q.dtype,
        device="cuda",
        max_q_tokens=257,
        max_kv_tokens=257,
        config=config,
    )
    runner = StreamingAttentionRunner(plan, config)
    runner(q, k, v, cu, cu)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(20):
        runner(q, k, v, cu, cu)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline + 2 * 2**20


def test_device_consumer_q_reuse_matches_separate_output_path():
    torch.manual_seed(29)
    q = torch.randn(83, 4, 32, dtype=torch.bfloat16, pin_memory=True)
    k = torch.randn(91, 2, 32, dtype=torch.bfloat16, pin_memory=True)
    v = torch.randn_like(k).pin_memory()
    cu_q = torch.tensor([0, 37, 83], dtype=torch.int32)
    cu_k = torch.tensor([0, 41, 91], dtype=torch.int32)

    def identity_consumer(attention, start, stop):
        del start, stop
        return attention.reshape(-1, 4, 32)

    outputs = []
    runners = []
    for output_mode in ("host", "device_consumer"):
        config = StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=31,
            kv_chunk_tokens=32,
            block_m=16,
            block_n=16,
            num_output_buffers=2,
            output_mode=output_mode,
        )
        plan = build_plan(
            q_heads=4,
            kv_heads=2,
            head_dim=32,
            dtype=q.dtype,
            device="cuda",
            max_q_tokens=q.shape[0],
            max_kv_tokens=k.shape[0],
            config=config,
        )
        runner = StreamingAttentionRunner(plan, config)
        out = torch.empty_like(q, pin_memory=True)
        runner.run_with_device_output(
            q,
            k,
            v,
            cu_q,
            cu_k,
            output_transform=identity_consumer,
            out=out,
            causal=True,
        )
        outputs.append(out)
        runners.append(runner)

    torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
    assert len(runners[0]._workspace.output) == 2
    assert runners[1]._workspace.output == []
