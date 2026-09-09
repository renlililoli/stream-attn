"""Exercise a standalone executable in a temporary working directory over real HTTP."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--python-launcher", type=Path)
    args = parser.parse_args()
    command = [str(args.executable.resolve())]
    if args.python_launcher:
        command += ["-S", str(args.python_launcher.resolve())]
    command += ["--no-browser", "--port", "0"]
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    with tempfile.TemporaryDirectory(prefix="estimator smoke ") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                deadline = time.monotonic() + 60
                url = None
                while time.monotonic() < deadline:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"H3 simulator: (http://127\.0\.0\.1:\d+)", text)
                    if match:
                        url = match.group(1)
                        break
                    if process.poll() is not None:
                        raise RuntimeError(f"Application exited: {text}")
                    time.sleep(0.1)
                if url is None:
                    raise RuntimeError(f"Startup timeout: {log_path.read_text(errors='replace')}")
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

                def request(path, payload=None):
                    data = None if payload is None else json.dumps(payload).encode()
                    req = urllib.request.Request(
                        url + path, data=data, headers={"Content-Type": "application/json"}
                    )
                    with opener.open(req, timeout=30) as response:
                        return response.read()

                assert json.loads(request("/health"))["status"] == "ok"
                assert "H3 Block 仿真台" in request("/").decode()
                assert json.loads(request("/api/schema"))["fields"]
                result = json.loads(
                    request(
                        "/api/simulate",
                        {
                            "parameters": {
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
                                "callback_variant": "modulated",
                                "execution_mode": "compare",
                            }
                        },
                    )
                )
                assert result["candidate_count"] == 2
                assert len(result["summary"]) == 2
                assert all(item["seconds"] > 0 for item in result["summary"])
                assert "<html" in result["html"].lower()
                print("PASS: standalone startup, page, schema, both H3 modes, HTML report")
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
