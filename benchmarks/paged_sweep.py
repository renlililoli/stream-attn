from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the seqattn paged benchmark matrix")
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[61_312, 132_288, 262_144, 1_048_576],
    )
    parser.add_argument("--host-budget-gib", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--workspace-gib", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--storage", nargs="+", default=["memory", "nvme-bf16", "nvme-int8"])
    parser.add_argument("--page-mib", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--queue-depth", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    matrix = itertools.product(
        args.tokens,
        args.host_budget_gib,
        args.workspace_gib,
        args.storage,
        args.page_mib,
        args.queue_depth,
    )
    for tokens, host, workspace, storage, page, queue_depth in matrix:
        name = f"{storage}_n{tokens}_host{host}_hbm{workspace}_p{page}_q{queue_depth}"
        output = args.output_dir / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "seqattn.paged_benchmark",
            "--tokens",
            str(tokens),
            "--host-budget-gib",
            str(host),
            "--workspace-gib",
            str(workspace),
            "--storage",
            storage,
            "--page-mib",
            str(page),
            "--queue-depth",
            str(queue_depth),
            "--output",
            str(output),
            *args.extra,
        ]
        completed = subprocess.run(command, check=False)
        row = json.loads(output.read_text()) if output.exists() else {
            "status": "runtime_error",
            "failure_message": f"benchmark exited {completed.returncode}",
        }
        summary.append({"name": name, **row})
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
