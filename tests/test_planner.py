import pytest
import torch

from seqattn_core import StreamingAttentionConfig, build_plan
from seqattn_core.kernels import profiles as kernel_profiles
from seqattn_core.planner import estimate_workspace_bytes


def test_planner_uses_largest_query_chunk_that_fits_budget():
    shape = {
        "q_heads": 16,
        "kv_heads": 4,
        "head_dim": 64,
        "dtype": torch.bfloat16,
        "num_kv_buffers": 2,
        "num_output_buffers": 2,
    }
    budget = estimate_workspace_bytes(q_tokens=512, kv_tokens=256, **shape)
    plan = build_plan(
        q_heads=shape["q_heads"],
        kv_heads=shape["kv_heads"],
        head_dim=shape["head_dim"],
        dtype=shape["dtype"],
        device="cpu",
        max_q_tokens=2048,
        max_kv_tokens=2048,
        config=StreamingAttentionConfig(
            workspace_budget_bytes=budget,
            kv_chunk_tokens=256,
            num_output_buffers=2,
            backend="reference",
        ),
    )
    assert plan.q_chunk_tokens == 512
    assert plan.estimated_workspace_bytes <= budget
    assert plan.group_size == 4


def test_planner_rejects_invalid_gqa_ratio():
    with pytest.raises(ValueError, match="multiple"):
        build_plan(
            q_heads=6,
            kv_heads=4,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cpu",
            max_q_tokens=16,
            max_kv_tokens=16,
        )


def test_planner_rejects_explicit_chunks_over_budget():
    with pytest.raises(ValueError, match="exceeding"):
        build_plan(
            q_heads=16,
            kv_heads=4,
            head_dim=64,
            dtype=torch.bfloat16,
            device="cpu",
            max_q_tokens=2048,
            max_kv_tokens=2048,
            config=StreamingAttentionConfig(
                workspace_budget_bytes=40 * 2**20,
                q_chunk_tokens=2048,
                kv_chunk_tokens=1024,
                backend="reference",
            ),
        )


def test_joint_planner_prefers_8k_kv_and_large_resident_q_for_h3_shape():
    plan = build_plan(
        q_heads=56,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cpu",
        max_q_tokens=61_312,
        max_kv_tokens=61_312,
        config=StreamingAttentionConfig(
            workspace_budget_bytes=2 * 2**30,
            output_mode="device_consumer",
            backend="reference",
        ),
    )
    assert plan.kv_chunk_tokens == 8192
    assert 43_000 <= plan.q_chunk_tokens <= 50_000
    assert plan.estimated_workspace_bytes <= 2 * 2**30


def test_device_consumer_plan_does_not_charge_raw_output_buffer():
    shape = {
        "q_tokens": 512,
        "kv_tokens": 256,
        "q_heads": 16,
        "kv_heads": 4,
        "head_dim": 64,
        "dtype": torch.bfloat16,
        "num_kv_buffers": 2,
        "num_output_buffers": 2,
    }
    host_output = estimate_workspace_bytes(**shape, output_mode="host")
    device_consumer = estimate_workspace_bytes(**shape, output_mode="device_consumer")
    expected_output_bytes = 2 * 512 * 16 * 64 * 2
    assert host_output - device_consumer == expected_output_bytes


def test_default_kernel_profile_is_portable_on_cpu():
    plan = build_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cpu",
        max_q_tokens=1024,
        max_kv_tokens=1024,
    )
    assert (plan.block_m, plan.block_n, plan.num_warps, plan.num_stages) == (64, 64, 4, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_default_kernel_profile_uses_blackwell_d128_preset():
    plan = build_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        max_q_tokens=1024,
        max_kv_tokens=1024,
    )
    major, _ = torch.cuda.get_device_capability()
    if major >= 12:
        expected = (128, 64, 8, 3)
    elif (
        major == 8
        and "A30" in torch.cuda.get_device_name().upper()
        and kernel_profiles.triton_major_minor() == (3, 7)
    ):
        expected = (128, 64, 8, 4)
    else:
        expected = (64, 64, 4, 2)
    assert (plan.block_m, plan.block_n, plan.num_warps, plan.num_stages) == expected


def test_default_kernel_profile_uses_a30_triton37_d128_preset(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA A30")
    monkeypatch.setattr(kernel_profiles, "triton_major_minor", lambda: (3, 7))
    plan = build_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda:0",
        max_q_tokens=1024,
        max_kv_tokens=1024,
    )
    assert (plan.block_m, plan.block_n, plan.num_warps, plan.num_stages) == (128, 64, 8, 4)


@pytest.mark.parametrize("triton_version", [(3, 3), (3, 8), None])
def test_a30_d128_preset_falls_back_outside_triton37(monkeypatch, triton_version):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA A30")
    monkeypatch.setattr(kernel_profiles, "triton_major_minor", lambda: triton_version)
    plan = build_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cuda:0",
        max_q_tokens=1024,
        max_kv_tokens=1024,
    )
    assert (plan.block_m, plan.block_n, plan.num_warps, plan.num_stages) == (64, 64, 4, 2)


def test_explicit_kernel_parameter_uses_portable_defaults_for_the_rest():
    plan = build_plan(
        q_heads=8,
        kv_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
        device="cpu",
        max_q_tokens=1024,
        max_kv_tokens=1024,
        config=StreamingAttentionConfig(block_m=32),
    )
    assert (plan.block_m, plan.block_n, plan.num_warps, plan.num_stages) == (32, 64, 4, 2)
