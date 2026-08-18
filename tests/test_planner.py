import pytest
import torch

from seqattn import StreamingAttentionConfig, build_plan
from seqattn.planner import estimate_workspace_bytes


def test_planner_uses_largest_query_chunk_that_fits_budget():
    shape = dict(
        q_heads=16,
        kv_heads=4,
        head_dim=64,
        dtype=torch.bfloat16,
        num_kv_buffers=2,
        num_output_buffers=2,
    )
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
