import gzip
import http.client
import json
import threading
from contextlib import contextmanager

import pytest

from seqattn_core.estimation import build_h3_block_execution, estimate_activation_memory
from seqattn_core.estimation.report import render_timeline_html, timeline_report_data
from seqattn_core.estimation.web.inputs import prepare_request, simulate
from seqattn_core.estimation.web.server import SimulationServer

SMALL = {
    "tokens": 14,
    "segments": "9,5",
    "hidden_features": 32,
    "ffn_features": 64,
    "heads": 4,
    "head_dim": 8,
    "rope_dim": 8,
    "q_chunk_tokens": 4,
    "kv_tile_tokens": 3,
    "projection_tile_tokens": 4,
    "ffn_tile_tokens": 8,
    "linear_memory": "dense",
    "callback_variant": "block25",
}


@contextmanager
def server():
    instance = SimulationServer(("127.0.0.1", 0))
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(2)


def request(instance, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", instance.server_port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        if response.getheader("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
        return response.status, dict(response.getheaders()), payload
    finally:
        connection.close()


def test_web_simulation_is_exactly_the_python_engine_result():
    parameters = {**SMALL, "execution_mode": "compare", "ffn_candidates": "4,8"}
    normalized, shape, configs, profile, callbacks, weights, _ = prepare_request(parameters)
    direct = estimate_activation_memory(
        [
            build_h3_block_execution(shape, c, profile, callbacks=callbacks, weights=weights)
            for c in configs
        ],
        objective_pool=profile.device_pool.name,
        objective_owners=frozenset({"operator", "callback", "caller"}),
        target_throughput_fraction=0.95,
    )
    web = simulate(parameters)
    assert web["parameters"] == normalized
    assert web["report"] == timeline_report_data(direct)
    assert len(web["report"]["candidates"]) == 4
    assert web["parameters"]["tokens"] == 14


def test_web_byte_units_and_capacity_constraints():
    result = simulate({**SMALL, "device_capacity_gib": 0.000001, "projection_weight_mib": 1})
    assert result["report"]["selection"]["selected_index"] is None
    _, _, _, profile, _, weights, _ = prepare_request(
        {**SMALL, "device_capacity_gib": 2, "projection_weight_mib": 3}
    )
    assert profile.device_pool.capacity_bytes == 2 * 2**30
    assert weights.projection_bytes == 3 * 2**20


def test_imported_profile_remains_authoritative_and_bound():
    _, shape, _, profile, _, _, _ = prepare_request(SMALL)
    imported = profile.for_shape(shape).to_dict()
    slow = simulate({**SMALL, "fa_tflops": 1}, imported)
    fast = simulate({**SMALL, "fa_tflops": 1000}, imported)
    assert slow["profile_imported"] is True
    assert slow["report"] == fast["report"]
    with pytest.raises(ValueError, match="signature"):
        simulate({**SMALL, "heads": 8}, imported)


@pytest.mark.parametrize(
    "bad",
    [
        {"tokens": True},
        {"fa_tflops": float("nan")},
        {"h2d_gbps": 0},
        {"tokens": 1.5},
        {"heads": "4"},
        {"execution_mode": "unknown"},
        {"typo": 12},
        {"segments": "2,-3"},
    ],
)
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        simulate({**SMALL, **bad})


def test_large_graph_and_candidate_count_are_rejected_before_model_construction(monkeypatch):
    import seqattn_core.estimation.web.inputs as module

    monkeypatch.setattr(
        module,
        "build_h3_block_execution",
        lambda *a, **k: pytest.fail("must reject before building"),
    )
    with pytest.raises(ValueError, match="交互上限"):
        simulate(
            {**SMALL, "segments": "", "tokens": 1000000, "q_chunk_tokens": 1, "kv_tile_tokens": 1}
        )
    with pytest.raises(ValueError, match="候选"):
        simulate({**SMALL, "ffn_candidates": ",".join(str(x) for x in range(1, 18))})


def test_web_routes_errors_compression_and_no_arbitrary_file_serving():
    with server() as instance:
        status, headers, body = request(instance, "GET", "/", headers={"Accept-Encoding": "gzip"})
        assert status == 200 and headers["Content-Encoding"] == "gzip"
        assert "H3 Block 仿真台" in body.decode()
        status, _, body = request(instance, "GET", "/api/schema")
        assert status == 200
        assert any(f["name"] == "fa_tflops" for f in json.loads(body)["fields"])
        assert request(instance, "GET", "/../../pyproject.toml")[0] == 404
        assert (
            request(instance, "POST", "/api/simulate", "{}", {"Content-Type": "text/plain"})[0]
            == 415
        )
        assert (
            request(
                instance, "POST", "/api/simulate", "{bad", {"Content-Type": "application/json"}
            )[0]
            == 400
        )
        assert (
            request(
                instance,
                "POST",
                "/api/simulate",
                '{"parameters":{"fa_tflops":NaN}}',
                {"Content-Type": "application/json"},
            )[0]
            == 400
        )
        assert (
            request(
                instance,
                "POST",
                "/api/simulate",
                "{}",
                {"Content-Type": "application/json", "Origin": "https://other.invalid"},
            )[0]
            == 403
        )
        status, _, body = request(
            instance,
            "POST",
            "/api/simulate",
            json.dumps({"parameters": {**SMALL, "h2d_gbps": 0}}),
            {"Content-Type": "application/json"},
        )
        assert status == 422 and "error" in json.loads(body)
        status, _, body = request(
            instance,
            "POST",
            "/api/simulate",
            json.dumps({"parameters": SMALL}),
            {"Content-Type": "application/json", "Accept-Encoding": "gzip"},
        )
        data = json.loads(body)
        assert status == 200 and data["candidate_count"] == 1
        assert data["summary"][0]["ffn_calls"] == 2
        assert data["html"] == render_timeline_html(simulate(SMALL)["report"])


def test_busy_server_rejects_extra_work_and_recovers():
    with server() as instance:
        instance.simulation_lock.acquire()
        try:
            status, _, body = request(
                instance,
                "POST",
                "/api/simulate",
                json.dumps({"parameters": SMALL}),
                {"Content-Type": "application/json"},
            )
            assert status == 429 and json.loads(body)["retry_ms"] > 0
        finally:
            instance.simulation_lock.release()
        assert request(instance, "GET", "/health")[0] == 200


@pytest.mark.parametrize(
    "bad_profile",
    [
        {"operators": []},
        {"operators": {}, "local_pools": [{}] * 9},
        {"operators": {"fc1": {"additional_workspaces": [{}] * 5}}},
    ],
)
def test_imported_profile_interactive_limits_are_validated(bad_profile):
    with pytest.raises(ValueError, match="Profile"):
        prepare_request(SMALL, bad_profile)


def test_desktop_launcher_opens_actual_bound_port(monkeypatch, capsys):
    from seqattn_core.estimation.web import server as module

    opened = []
    ready = threading.Event()

    def browser(url):
        opened.append(url)
        ready.set()
        return True

    def serve(instance):
        assert ready.wait(5)
        assert opened == [f"http://127.0.0.1:{instance.server_port}"]
        assert instance.server_port > 0
        raise KeyboardInterrupt

    monkeypatch.setattr(module.webbrowser, "open", browser)
    monkeypatch.setattr(module.SimulationServer, "serve_forever", serve)
    module.main([], default_port=0, default_open_browser=True)
    assert opened[0] in capsys.readouterr().out


def test_desktop_launcher_can_disable_browser(monkeypatch):
    from seqattn_core.estimation.web import server as module

    monkeypatch.setattr(module.webbrowser, "open", lambda url: pytest.fail("browser disabled"))

    def serve(instance):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.SimulationServer, "serve_forever", serve)
    module.main(["--no-browser"], default_port=0, default_open_browser=True)


def test_imported_profile_controls_packing_concurrency():
    from dataclasses import replace

    _, shape, _, profile, _, _, _ = prepare_request(SMALL)
    imported = replace(profile, projection_pack_resources=None).for_shape(shape).to_dict()
    result = simulate({**SMALL, "projection_pack_mode": "concurrent"}, imported)
    operations = result["report"]["candidates"][0]["trace"]["operations"]
    assert all(o["resources"] == ("compute",) for o in operations if o["name"].endswith(".pack"))


def test_production_web_path_uses_actual_combined_fc2_entry_point():
    _, _, _, profile, callbacks, _, _ = prepare_request(
        {**SMALL, "callback_variant": "modulated", "fc2_tflops": 17}
    )
    assert callbacks.fused_swiglu_fc2
    assert profile.operators["swiglu_fc2"].rate.work_per_second == 17e12
