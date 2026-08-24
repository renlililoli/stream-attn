from __future__ import annotations

import torch

from seqattn_core.streaming.flash_backends import Flash3Backend, Flash4Backend


def test_fa3_adapter_normalizes_public_output(monkeypatch):
    output = torch.empty(1, 5, 3, 8)
    expected = torch.randn_like(output)
    lse = torch.randn(1, 3, 5, dtype=torch.float32)

    def fake_flash3(q, k, v, **kwargs):
        del q, k, v
        assert kwargs["return_attn_probs"] is True
        return expected, lse

    monkeypatch.setattr(Flash3Backend, "load_function", staticmethod(lambda: fake_flash3))
    actual, actual_lse = Flash3Backend().forward_partial(
        output,
        output,
        output,
        output,
        softmax_scale=0.5,
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, lse)


def test_fa4_adapter_normalizes_transposed_lse(monkeypatch):
    output = torch.empty(1, 5, 3, 8)
    expected = torch.randn_like(output)
    lse = torch.randn(1, 5, 3, dtype=torch.float32)

    def fake_flash4(q, k, v, **kwargs):
        del q, k, v
        assert kwargs["return_lse"] is True
        return expected, lse

    monkeypatch.setattr(Flash4Backend, "load_function", staticmethod(lambda: fake_flash4))
    actual, actual_lse = Flash4Backend().forward_partial(
        output,
        output,
        output,
        output,
        softmax_scale=0.5,
    )
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lse, lse.transpose(1, 2))
