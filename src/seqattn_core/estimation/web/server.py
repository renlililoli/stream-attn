"""Loopback-first local HTTP UI for the CPU-only H3 estimator."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from ..report import render_timeline_html
from .inputs import schema, simulate
from .page import PAGE

MAX_BODY_BYTES = 2 * 2**20
LOG = logging.getLogger(__name__)


class SimulationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, *, simulator=simulate):
        super().__init__(address, Handler)
        self.simulator = simulator
        self.simulation_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: SimulationServer
    server_version = "SeqAttnLocal/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, message, *args):
        LOG.info("%s %s", self.address_string(), message % args)

    def _reply(self, status, data, content_type="application/json; charset=utf-8"):
        payload = (
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            if not isinstance(data, bytes)
            else data
        )
        compressed = len(payload) > 1024 and "gzip" in self.headers.get("Accept-Encoding", "")
        if compressed:
            payload = gzip.compress(payload, compresslevel=3)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _origin_ok(self):
        host = self.headers.get("Host", "")
        try:
            hostname = urlsplit("http://" + host).hostname
        except ValueError:
            return False
        if self.server.server_address[0] in {"127.0.0.1", "::1"} and hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {"http://" + host, "https://" + host}

    def do_GET(self):
        if not self._origin_ok():
            self._reply(HTTPStatus.FORBIDDEN, {"error": "请从本服务页面访问。"})
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._reply(HTTPStatus.OK, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/schema":
            self._reply(HTTPStatus.OK, schema())
        elif path == "/health":
            self._reply(HTTPStatus.OK, {"status": "ok", "engine": "seqattn_core.estimation.h3"})
        elif path == "/favicon.ico":
            self._reply(HTTPStatus.NO_CONTENT, b"")
        else:
            self._reply(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/simulate":
            self._reply(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not self._origin_ok():
            self._reply(HTTPStatus.FORBIDDEN, {"error": "不接受跨站仿真请求。"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请使用 application/json。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY_BYTES:
            self._reply(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求必须非空且不超过 2 MiB。"}
            )
            return
        try:

            def invalid_constant(value):
                raise ValueError(f"不支持的数值：{value}")

            payload = json.loads(self.rfile.read(length), parse_constant=invalid_constant)
            if not isinstance(payload, dict) or set(payload) - {"parameters", "profile"}:
                raise ValueError("请求需要 parameters 对象和可选的 profile。")
            parameters = payload.get("parameters", {})
        except (ValueError, TypeError, UnicodeDecodeError, RecursionError, TimeoutError) as error:
            self._reply(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if not self.server.simulation_lock.acquire(blocking=False):
            self._reply(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "仿真正在运行，请稍后重试。", "retry_ms": 500},
            )
            return
        started = time.perf_counter()
        try:
            result = self.server.simulator(parameters, payload.get("profile"))
            report = result.pop("report")
            result["html"] = render_timeline_html(report)
            result["elapsed_ms"] = (time.perf_counter() - started) * 1000
            result["candidate_count"] = len(report["candidates"])
            result["selected_index"] = report["selection"]["selected_index"]
            result["summary"] = [
                {
                    "name": item["trace"]["name"],
                    "seconds": item["trace"]["duration_seconds"],
                    "activation_peak_bytes": item.get("objective_peak_bytes"),
                    "ffn_calls": len(item["trace"]["metadata"]["ffn_ranges"]),
                    "fits_capacity": all(
                        p["capacity_bytes"] is None or p["peak_bytes"] <= p["capacity_bytes"]
                        for p in item["stats"]["pools"]
                    ),
                }
                for item in report["candidates"]
            ]
            self._reply(HTTPStatus.OK, result)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            self._reply(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
        except Exception:
            LOG.exception("Simulation failed")
            self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "仿真失败，请查看服务日志。"})
        finally:
            self.server.simulation_lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with SimulationServer((args.host, args.port)) as server:
        print(f"H3 simulator: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
