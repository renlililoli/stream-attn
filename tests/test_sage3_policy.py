import pytest
import torch

from seqattn_core.streaming import backend


def setup_sm120(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    monkeypatch.setattr(backend, "backend_is_available", lambda name: True)


def test_nvfp4_default_is_scoped_to_opted_in_container_and_sm120(monkeypatch):
    setup_sm120(monkeypatch)
    device = torch.device("cuda")
    monkeypatch.delenv("SEQATTN_AUTO_NVFP4", raising=False)
    assert backend.automatic_backend_order(device)[0] == "triton"
    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    assert (
        backend.resolve_backend(
            "auto", torch.bfloat16, device, head_dim=128, q_heads=56, kv_heads=56
        )
        == "sage3"
    )
    assert backend.resolve_backend("triton", torch.bfloat16, device, head_dim=128) == "triton"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    assert backend.automatic_backend_order(device)[0] == "fa4"


def test_unsupported_geometry_uses_safe_auto_fallback(monkeypatch):
    setup_sm120(monkeypatch)
    monkeypatch.setenv("SEQATTN_AUTO_NVFP4", "1")
    assert (
        backend.resolve_backend("auto", torch.bfloat16, torch.device("cuda"), head_dim=256)
        == "triton"
    )
    assert (
        backend.resolve_backend(
            "auto", torch.bfloat16, torch.device("cuda"), head_dim=128, q_heads=4, kv_heads=2
        )
        == "triton"
    )
    with pytest.raises(ValueError, match="equal Q/KV heads"):
        backend.resolve_backend(
            "sage3", torch.bfloat16, torch.device("cuda"), head_dim=128, q_heads=4, kv_heads=2
        )


def test_offline_h3_estimator_does_not_misrepresent_nvfp4_plan():
    from types import SimpleNamespace

    from seqattn_core.estimation import H3ExecutionConfig

    with pytest.raises(ValueError, match="not Sage3"):
        H3ExecutionConfig.from_attention_plan(
            SimpleNamespace(backend="sage3", backend_workspace_bytes=1),
            projection_tile_tokens=4096,
            ffn_tile_tokens=4096,
        )
