import pytest
import torch

from seqattn_core.kernels import (
    initialize_split_attention_state,
    merge_split_attention_state,
    triton_is_available,
)

pytestmark = pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")


def test_split_attention_state_matches_logsumexp_reference():
    torch.manual_seed(41)
    shape = (2, 19, 3, 32)
    partial_a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    partial_b = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    lse_a = torch.randn((2, 3, 19), device="cuda", dtype=torch.float32)
    lse_b = torch.randn((2, 3, 19), device="cuda", dtype=torch.float32)
    lse_a[0, 0, 0] = -torch.inf
    lse_b[0, 0, 0] = -torch.inf
    state_output = torch.empty(shape, device="cuda", dtype=torch.float32)
    state_lse = torch.empty((2, 19, 3), device="cuda", dtype=torch.float32)

    initialize_split_attention_state(partial_a, lse_a, state_output, state_lse)
    merge_split_attention_state(partial_b, lse_b, state_output, state_lse)
    torch.cuda.synchronize()

    lse_a_rows = lse_a.transpose(1, 2)
    lse_b_rows = lse_b.transpose(1, 2)
    expected_lse = torch.logaddexp(lse_a_rows, lse_b_rows)
    weight_a = torch.exp(lse_a_rows - expected_lse).nan_to_num()
    weight_b = torch.exp(lse_b_rows - expected_lse).nan_to_num()
    expected_output = (
        weight_a.unsqueeze(-1) * partial_a.float() + weight_b.unsqueeze(-1) * partial_b.float()
    )

    torch.testing.assert_close(state_lse, expected_lse, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state_output, expected_output, atol=2e-5, rtol=2e-5)
