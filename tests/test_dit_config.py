from __future__ import annotations

import pytest

from seqattn_core.dit.ltx2 import LTX2Config, load_ltx2_config
from seqattn_core.dit.minimax_h3 import H3Config, load_h3_config
from seqattn_core.dit.wan import WanConfig, load_wan_config


def test_all_model_sections_can_share_one_config_file(tmp_path):
    path = tmp_path / "seqattn.toml"
    path.write_text(
        "[attention]\nbackend = 'auto'\n\n"
        "[minimax_h3]\n"
        "execution_mode = 'recompute'\n"
        "projection_tile_tokens = 4096\n"
        "ffn_tile_tokens = 3072\n\n"
        "[wan]\n"
        "execution_mode = 'materialized'\n"
        "projection_tile_tokens = 1536\n"
        "ffn_tile_tokens = 1024\n\n"
        "[ltx2]\n"
        "execution_mode = 'recompute'\n"
        "projection_tile_tokens = 1024\n"
        "video_ffn_tile_tokens = 2048\n"
        "audio_ffn_tile_tokens = 768\n\n"
        "[consumer_specific]\nenabled = true\n",
        encoding="utf-8",
    )

    assert load_h3_config(path) == H3Config(
        execution_mode="recompute",
        projection_tile_tokens=4096,
        ffn_tile_tokens=3072,
    )
    assert load_wan_config(path) == WanConfig(
        execution_mode="materialized",
        projection_tile_tokens=1536,
        ffn_tile_tokens=1024,
    )
    assert load_ltx2_config(path) == LTX2Config(
        execution_mode="recompute",
        projection_tile_tokens=1024,
        video_ffn_tile_tokens=2048,
        audio_ffn_tile_tokens=768,
    )


def test_seqattn_config_environment_is_used(tmp_path, monkeypatch):
    path = tmp_path / "seqattn.toml"
    path.write_text("[wan]\nffn_tile_tokens = 640\n", encoding="utf-8")
    monkeypatch.setenv("SEQATTN_CONFIG", str(path))

    assert load_wan_config().ffn_tile_tokens == 640


def test_explicit_path_overrides_seqattn_config_environment(tmp_path, monkeypatch):
    environment_path = tmp_path / "environment.toml"
    explicit_path = tmp_path / "explicit.toml"
    environment_path.write_text("[wan]\nffn_tile_tokens = 640\n", encoding="utf-8")
    explicit_path.write_text("[wan]\nffn_tile_tokens = 896\n", encoding="utf-8")
    monkeypatch.setenv("SEQATTN_CONFIG", str(environment_path))

    assert load_wan_config(explicit_path).ffn_tile_tokens == 896


def test_missing_default_config_uses_model_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQATTN_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert load_h3_config() == H3Config()
    assert load_wan_config() == WanConfig()
    assert load_ltx2_config() == LTX2Config()


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_missing_selected_config_is_an_error(tmp_path, monkeypatch, source):
    path = tmp_path / "missing.toml"
    if source == "environment":
        monkeypatch.setenv("SEQATTN_CONFIG", str(path))
        load = load_wan_config
    else:
        load = lambda: load_wan_config(path)

    with pytest.raises(FileNotFoundError, match="SeqAttn config does not exist"):
        load()


@pytest.mark.parametrize("table", ["minimax_h3", "wan", "ltx2"])
def test_model_section_must_be_a_table(tmp_path, table):
    path = tmp_path / "seqattn.toml"
    path.write_text(f"{table} = ['invalid']\n", encoding="utf-8")

    loader = {
        "minimax_h3": load_h3_config,
        "wan": load_wan_config,
        "ltx2": load_ltx2_config,
    }[table]
    with pytest.raises(TypeError, match=rf"\[{table}\].*TOML table"):
        loader(path)


@pytest.mark.parametrize(
    ("table", "key", "value", "loader"),
    [
        ("minimax_h3", "projection_tile_tokens", "true", load_h3_config),
        ("minimax_h3", "ffn_tile_tokens", "0", load_h3_config),
        ("wan", "projection_tile_tokens", "'2048'", load_wan_config),
        ("wan", "ffn_tile_tokens", "-1", load_wan_config),
        ("ltx2", "projection_tile_tokens", "false", load_ltx2_config),
        ("ltx2", "video_ffn_tile_tokens", "0", load_ltx2_config),
        ("ltx2", "audio_ffn_tile_tokens", "1.5", load_ltx2_config),
    ],
)
def test_model_tile_values_must_be_positive_integers(tmp_path, table, key, value, loader):
    path = tmp_path / "seqattn.toml"
    path.write_text(f"[{table}]\n{key} = {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{table}\.{key}.*positive integer"):
        loader(path)


@pytest.mark.parametrize(
    ("table", "loader"),
    [
        ("minimax_h3", load_h3_config),
        ("wan", load_wan_config),
        ("ltx2", load_ltx2_config),
    ],
)
@pytest.mark.parametrize("value", ["'fallback'", "true", "['recompute']"])
def test_execution_mode_is_strict(tmp_path, table, loader, value):
    path = tmp_path / "seqattn.toml"
    path.write_text(f"[{table}]\nexecution_mode = {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{table}\.execution_mode"):
        loader(path)


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
def test_ltx2_supports_both_execution_modes(tmp_path, mode):
    path = tmp_path / "seqattn.toml"
    path.write_text(f"[ltx2]\nexecution_mode = '{mode}'\n", encoding="utf-8")

    assert load_ltx2_config(path).execution_mode == mode


@pytest.mark.parametrize(
    ("table", "key", "loader"),
    [
        ("minimax_h3", "qkv_tile_tokens", load_h3_config),
        ("minimax_h3", "mlp_tile_tokens", load_h3_config),
        ("wan", "qkv_tile_tokens", load_wan_config),
        ("ltx2", "ffn_tile_tokens", load_ltx2_config),
    ],
)
def test_model_sections_reject_unknown_or_legacy_keys(tmp_path, table, key, loader):
    path = tmp_path / "seqattn.toml"
    path.write_text(f"[{table}]\n{key} = 1024\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"\[{table}\].*unknown keys.*{key}"):
        loader(path)


def test_dataclass_validation_rejects_invalid_direct_construction():
    with pytest.raises(ValueError, match="minimax_h3.execution_mode"):
        H3Config(execution_mode="fallback")
    with pytest.raises(ValueError, match="wan.projection_tile_tokens"):
        WanConfig(projection_tile_tokens=True)
    with pytest.raises(ValueError, match="ltx2.audio_ffn_tile_tokens"):
        LTX2Config(audio_ffn_tile_tokens=0)
