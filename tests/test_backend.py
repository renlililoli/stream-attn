from __future__ import annotations

import torch

from seqattn_core.streaming import backend as backend_module
from seqattn_core.streaming.backend import configured_backend_name, resolve_backend


def _mock_cuda(monkeypatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)


def _mock_available(monkeypatch, names: set[str]) -> None:
    monkeypatch.setattr(
        backend_module,
        "backend_is_available",
        lambda name: name in names,
    )


def test_sm80_auto_prefers_fa2(monkeypatch):
    _mock_cuda(monkeypatch, (8, 0))
    _mock_available(monkeypatch, {"fa2", "triton", "reference"})
    assert resolve_backend("auto", torch.bfloat16, torch.device("cuda:0"), head_dim=128) == "fa2"


def test_sm80_auto_falls_back_to_builtin(monkeypatch):
    _mock_cuda(monkeypatch, (8, 6))
    _mock_available(monkeypatch, {"triton", "reference"})
    assert (
        resolve_backend("auto", torch.float16, torch.device("cuda:0"), head_dim=128)
        == "triton"
    )


def test_sm90_uses_fa3_and_sm120_preserves_builtin_default(monkeypatch):
    _mock_available(monkeypatch, {"fa2", "fa3", "fa4", "triton", "reference"})
    _mock_cuda(monkeypatch, (9, 0))
    assert resolve_backend("auto", torch.bfloat16, torch.device("cuda:0")) == "fa3"
    _mock_cuda(monkeypatch, (12, 0))
    assert resolve_backend("auto", torch.bfloat16, torch.device("cuda:0")) == "triton"


def test_sm120_allows_explicit_fa4(monkeypatch):
    _mock_cuda(monkeypatch, (12, 0))
    _mock_available(monkeypatch, {"fa4", "triton", "reference"})
    assert resolve_backend("fa4", torch.bfloat16, torch.device("cuda:0")) == "fa4"


def test_sm120_falls_back_to_preserved_builtin_kernel(monkeypatch):
    _mock_cuda(monkeypatch, (12, 0))
    _mock_available(monkeypatch, {"triton", "reference"})
    assert resolve_backend("auto", torch.bfloat16, torch.device("cuda:0")) == "triton"


def test_builtin_and_legacy_flash2_names_are_compatible(monkeypatch):
    _mock_cuda(monkeypatch, (8, 0))
    _mock_available(monkeypatch, {"triton", "fa2"})
    device = torch.device("cuda:0")
    assert resolve_backend("builtin", torch.bfloat16, device) == "triton"
    assert resolve_backend("flash2_split", torch.bfloat16, device) == "fa2"


def test_explicit_backend_overrides_environment_and_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[attention]\nbackend = "fa4"\n', encoding="utf-8")
    monkeypatch.setenv("SEQATTN_CONFIG", str(config_path))
    monkeypatch.setenv("SEQATTN_BACKEND", "fa3")
    assert configured_backend_name("fa2") == "fa2"


def test_environment_overrides_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[attention]\nbackend = "fa4"\n', encoding="utf-8")
    monkeypatch.setenv("SEQATTN_CONFIG", str(config_path))
    monkeypatch.setenv("SEQATTN_BACKEND", "builtin")
    assert configured_backend_name(None) == "triton"


def test_toml_is_used_when_no_environment_override(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[attention]\nbackend = "fa2"\n', encoding="utf-8")
    monkeypatch.setenv("SEQATTN_CONFIG", str(config_path))
    monkeypatch.delenv("SEQATTN_BACKEND", raising=False)
    assert configured_backend_name(None) == "fa2"


def test_runtime_can_limit_auto_to_builtin(monkeypatch):
    _mock_cuda(monkeypatch, (8, 0))
    _mock_available(monkeypatch, {"fa2", "triton", "reference"})
    assert (
        resolve_backend(
            "auto",
            torch.bfloat16,
            torch.device("cuda:0"),
            allowed={"builtin", "reference"},
        )
        == "triton"
    )
