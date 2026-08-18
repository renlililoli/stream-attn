from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run each seqattn benchmark point in a new process"
    )
    parser.add_argument("--tokens", nargs="+", type=int, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("seqattn", "flash2", "sdpa"), default=["seqattn", "flash2"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for tokens in args.tokens:
        for mode in args.modes:
            output = args.output_dir / f"{mode}_{tokens}.json"
            command = [
                sys.executable,
                "-m",
                "seqattn.benchmarking.streaming",
                "--mode",
                mode,
                "--tokens",
                str(tokens),
                "--output",
                str(output),
                *args.extra,
            ]
            completed = subprocess.run(command, check=False)
            row = (
                json.loads(output.read_text())
                if output.exists()
                else {
                    "status": "runtime_error",
                    "failure_message": f"benchmark process exited {completed.returncode}",
                }
            )
            summary.append({"mode": mode, "tokens": tokens, **row})
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
