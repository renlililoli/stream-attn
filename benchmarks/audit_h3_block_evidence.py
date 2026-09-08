"""Inventory retained H3 block-25 results and check their mapping to the estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_profile(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        gpu = connection.execute("SELECT name, busLocation, uuid FROM TARGET_INFO_GPU").fetchone()
        env_row = connection.execute(
            "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name='DeviceEnvironment'"
        ).fetchone()
        # Export only version fields, never the complete captured process environment.
        environment = dict(item.split("=", 1) for item in env_row[0].split(";") if "=" in item)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        return {
            "source": path.name,
            "sha256": sha256(path),
            "gpu": {"name": gpu[0], "pci_bus": gpu[1], "uuid": gpu[2]},
            "environment": {
                key: environment.get(key)
                for key in ("CUDA_VERSION", "PYTORCH_VERSION", "NVIDIA_PYTORCH_VERSION")
            },
            "kernel_events": connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0],
            "memcpy_events": connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY"
            ).fetchone()[0],
            "has_allocation_events": bool(
                tables & {"CUDA_GPU_MEMORY_USAGE_EVENTS", "CUPTI_ACTIVITY_KIND_MEMORY"}
            ),
            "use": "hardware/environment and scheduling attribution; not primary latency",
        }


def inventory(directory):
    rows = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        c, plan = raw["configuration"], raw["plan"]
        times = [row["wall_seconds"] for row in raw["records"]]
        native = c.get("mode") == "native"
        record_stats = raw["records"][0].get("stats", {})
        tokens, q, ffn = c["tokens"], plan["q_chunk_tokens"], plan["mlp_tile_tokens"]
        template_calls = sum(
            math.ceil(min(q, tokens - start) / ffn) for start in range(0, tokens, q)
        )
        rows.append(
            {
                "source": path.name,
                "sha256": sha256(path),
                "configuration": c,
                "model": raw["model"],
                "plan": plan,
                "kind": "native" if native else "streaming",
                "profile_capture": c.get("cuda_profiler_capture", False),
                "wall_seconds": times,
                "wall_median_seconds": raw["summary"]["wall_median_seconds"],
                "wall_mean_seconds": raw["summary"]["wall_mean_seconds"],
                "wall_min_seconds": min(times),
                "wall_max_seconds": max(times),
                "wall_range_over_min_percent": 100 * (max(times) / min(times) - 1),
                "torch_peak_allocated_bytes": raw["summary"].get("torch_peak_allocated_bytes"),
                "torch_peak_reserved_bytes": raw["summary"].get("torch_peak_reserved_bytes"),
                "native_cuda_stage_seconds": [r.get("gpu_stage_seconds") for r in raw["records"]]
                if native
                else None,
                "native_cuda_stage_means": {
                    name: value
                    for name, value in raw["summary"].items()
                    if name.endswith("_mean_seconds") and name != "wall_mean_seconds"
                }
                if native
                else None,
                "observed_q_chunks": record_stats.get("projection", {})
                .get("attention", {})
                .get("q_chunks"),
                "observed_ffn_calls": record_stats.get("mlp_chunks"),
                "observed_ffn_cross_q_boundaries": record_stats.get("mlp_cross_q_boundaries"),
                "legacy_template_ffn_calls": None if native else template_calls,
                "expected_carry_ffn_calls": None if native else math.ceil(tokens / ffn),
            }
        )
    by_name = {row["source"]: row for row in rows}
    lower = by_name["block25_q3840_kv4096_qkv4096_mlp8192.json"]
    upper = by_name["block25_q3840_kv4096_qkv4096_mlp16384.json"]
    for key in ("q_chunk_tokens", "kv_chunk_tokens", "qkv_tile_tokens"):
        if lower["plan"][key] != upper["plan"][key]:
            raise ValueError(
                "incremental-memory comparison requires fixed attention/projection tiles"
            )
    if lower["model"] != upper["model"]:
        raise ValueError("incremental-memory comparison requires the same model shape")
    hidden = lower["model"]["hidden_features"]
    ffn_delta = upper["plan"]["mlp_tile_tokens"] - lower["plan"]["mlp_tile_tokens"]
    # In the initial template both FFN tiles exceed Q=3840. Each consumer
    # invocation processes at most Q tokens; only persistent output slots grow.
    template_delta = 2 * ffn_delta * hidden * 2
    measured_delta = upper["torch_peak_allocated_bytes"] - lower["torch_peak_allocated_bytes"]
    return {
        "schema_version": 1,
        "audit_date": "2026-09-08",
        "source_directory": "workspace/benchmarks/results/seqattn_single_block25_81159_20260825",
        "source_repository": "../..",
        "rows": rows,
        "profiles": [inspect_profile(path) for path in sorted(directory.glob("*.sqlite"))],
        "incremental_memory_mapping_check": {
            "lower_source": lower["source"],
            "upper_source": upper["source"],
            "measured_torch_peak_growth_bytes": measured_delta,
            "recorded_core_workspace_growth_bytes": upper["plan"]["estimated_workspace_bytes"]
            - lower["plan"]["estimated_workspace_bytes"],
            "legacy_missing_carry_growth_bytes": ffn_delta * hidden * 2,
            "legacy_template_output_slot_growth_bytes": template_delta,
            "incremental_growth_error_percent": 100 * (template_delta / measured_delta - 1),
            "interpretation": "Mapping mismatch: initial per-Q FFN template does not reproduce H3 carry batching. "
            "This is an incremental peak comparison, not an absolute activation-only error.",
        },
        "legacy_template": "initial per-Q template, superseded by estimation.h3",
        "current_h3_model": "src/seqattn_core/estimation/h3/pipeline.py",
        "audit_script_sha256": sha256(Path(__file__)),
        "limitations": [
            "Native unprofiled JSON contains independent CUDA-event stage measurements at full sequence shape.",
            "Profiled SQLite has CUDA 13.1/NGC metadata; do not silently reuse CUDA 12.8 calibration from the other experiment.",
            "Native stage rates are not a measured small-tile GEMM saturation curve.",
            "Torch peaks include all live allocator tensors; weight-only baselines and allocation event traces are absent.",
            "All raw repeats are retained, including the visibly variable config_qkv4096_mlp4096 run.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "json_files": len(result["rows"]),
                "profiles": result["profiles"],
                "memory_mapping": result["incremental_memory_mapping_check"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
