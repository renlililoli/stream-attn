from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from seqattn.benchmarking.common import atomic_json


COARSE_Q = (2048, 4096, 6144, 8192, 10240, 12288, 16384, 24576, 32768)
FINE_Q = (4224, 4736, 5248, 5504, 5760, 6272, 6784, 7296)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered RTX 5090 Q sweep")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round-start", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20260824)
    parser.add_argument("--q-values", type=int, nargs="*")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0 or args.round_start <= 0:
        raise ValueError("rounds and round-start must be positive")
    prediction = json.loads(args.prediction.read_text(encoding="ascii"))
    if not prediction.get("prediction_created_before_q_sweep"):
        raise ValueError("prediction artifact is not marked as pre-sweep")

    q_values = sorted(set(args.q_values or (COARSE_Q + FINE_Q)))
    if not q_values or any(q <= 0 or q % 128 for q in q_values):
        raise ValueError("all Q values must be positive multiples of 128")
    git_commit = command_output(["git", "rev-parse", "HEAD"])
    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "git_commit": git_commit,
        "prediction_path": str(args.prediction),
        "prediction_sha256": sha256(args.prediction),
        "q_values": q_values,
        "coarse_q_values": list(COARSE_Q),
        "fine_q_values": list(FINE_Q),
        "round_start": args.round_start,
        "rounds": args.rounds,
        "shuffle_seed": args.shuffle_seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": [],
    }
    atomic_json(manifest_path, manifest)
    results = manifest["results"]
    assert isinstance(results, list)

    for round_index in range(args.round_start, args.round_start + args.rounds):
        order = q_values.copy()
        random.Random(args.shuffle_seed + round_index).shuffle(order)
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for order_index, q_tokens in enumerate(order):
            output = round_dir / f"q_{q_tokens:05d}.json"
            log = round_dir / f"q_{q_tokens:05d}.stdout.log"
            if output.exists():
                existing = json.loads(output.read_text(encoding="ascii"))
                if existing.get("status") == "success":
                    results.append(
                        {
                            "round": round_index,
                            "order": order_index,
                            "q_tokens": q_tokens,
                            "output": str(output),
                            "status": "existing_success",
                        }
                    )
                    atomic_json(manifest_path, manifest)
                    continue

            compute_processes = command_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,name,used_memory",
                    "--format=csv,noheader",
                ]
            ).splitlines()
            command = [
                sys.executable,
                "-m",
                "seqattn_core.benchmarking.streaming",
                "--mode",
                "seqattn",
                "--tokens",
                "524288",
                "--segments",
                "1",
                "--q-heads",
                "56",
                "--kv-heads",
                "56",
                "--head-dim",
                "128",
                "--dtype",
                "bfloat16",
                "--q-chunk",
                str(q_tokens),
                "--kv-chunk",
                "4096",
                "--block-m",
                "128",
                "--block-n",
                "64",
                "--num-warps",
                "8",
                "--num-stages",
                "3",
                "--num-kv-buffers",
                "2",
                "--num-output-buffers",
                "1",
                "--workspace-mib",
                "4096",
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--cpu-workers",
                "32",
                "--cpu-chunk-tokens",
                "4096",
                "--seed",
                "0",
                "--skip-memory-probe",
                "--output",
                str(output),
            ]
            started = time.time()
            with log.open("w", encoding="ascii") as stdout:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            row: dict[str, object] = {
                "round": round_index,
                "order": order_index,
                "q_tokens": q_tokens,
                "command": command,
                "compute_processes_before": compute_processes,
                "started_unix_seconds": started,
                "elapsed_seconds": time.time() - started,
                "returncode": completed.returncode,
                "output": str(output),
                "stdout_log": str(log),
            }
            if output.exists():
                payload = json.loads(output.read_text(encoding="ascii"))
                row["status"] = payload.get("status", "unknown")
                row["mean_seconds"] = payload.get("mean_seconds")
                row["mean_compute_pipeline_seconds"] = payload.get(
                    "mean_compute_pipeline_seconds"
                )
                row["compute_pipeline_effective_tflops"] = payload.get(
                    "compute_pipeline_effective_tflops"
                )
            else:
                row["status"] = "missing_output"
            results.append(row)
            atomic_json(manifest_path, manifest)
            if completed.returncode != 0 and not args.continue_on_error:
                manifest["status"] = "failed"
                atomic_json(manifest_path, manifest)
                raise SystemExit(completed.returncode)

    manifest["status"] = "success"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
