"""Replay the retained 81,159-token H3 block using the full H3 execution model.

CUDA-event native stage rates and profile-derived copy bandwidth are inputs;
unprofiled JSON wall times are the comparison target. This rate-transfer check
is distinct from exact-shape operator calibration and reports that limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3OperatorProfile,
    MemoryPool,
    RateProfile,
    build_h3_block_execution,
    memory_statistics,
    schedule_execution,
    write_timeline_report,
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration(directory):
    native_path = directory / "block25_native_fa4.json"
    native = read(native_path)
    m = native["model"]
    tokens = native["configuration"]["tokens"]
    shape = H3BlockShape((tokens,), m["hidden_features"], 14336, m["heads"], m["head_dim"])
    h, a, f = shape.hidden_features, shape.attention_features, shape.ffn_features
    times = native["summary"]
    rates = {
        "attention": 4 * tokens * tokens * a / times["attention_mean_seconds"],
        "qkv": 2 * tokens * h * 3 * a / times["qkv_projection_mean_seconds"],
        "out": 2 * tokens * a * h / times["out_projection_mean_seconds"],
        "ffn": 6 * tokens * h * f / times["mlp_mean_seconds"],
    }
    sql = directory / "block25_q3840_kv4096_steady.sqlite"
    with sqlite3.connect(sql.resolve().as_uri() + "?mode=ro", uri=True) as c:
        copies = {
            kind: size / seconds
            for kind, size, seconds in c.execute(
                "SELECT copyKind,SUM(bytes),SUM(end-start)/1e9 FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
            )
        }
        vector_seconds = c.execute(
            'SELECT SUM(k.end-k.start)/1e9 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE s.value="_finalize_attention_kernel"'
        ).fetchone()[0]
        gpu = c.execute("SELECT name,uuid FROM TARGET_INFO_GPU").fetchone()
    profile_json = read(directory / "block25_q3840_kv4096_profile.json")
    vector_rate = tokens * a * len(profile_json["records"]) / vector_seconds

    def compute(label, rate):
        return RateProfile(
            label,
            rate,
            ("compute",),
            provenance="native full-shape CUDA-event rate transfer; not tile calibration",
        )

    profile = H3DeviceProfile.from_rates(
        "RTX 5090 H3 block25 rate-transfer diagnostic",
        device_pool=MemoryPool("GPU.HBM", allocation_alignment_bytes=512),
        host_pool=MemoryPool("host.DRAM"),
        attention=compute("native FA4 FLOP/s", rates["attention"]),
        gemm=compute("native MLP effective FLOP/s", rates["ffn"]),
        vector=RateProfile(
            "finalize proxy elements/s",
            vector_rate,
            ("compute",),
            provenance="profile-derived vector proxy",
        ),
        h2d=RateProfile(
            "H2D byte/s",
            copies[1],
            ("H2D",),
            kind="io",
            provenance="profile-derived copy bandwidth",
        ),
        d2h=RateProfile(
            "D2H byte/s",
            copies[2],
            ("D2H",),
            kind="io",
            provenance="profile-derived copy bandwidth",
        ),
        d2d=RateProfile(
            "D2D byte/s", copies[8], ("compute",), provenance="profile-derived copy bandwidth"
        ),
    )
    operators = dict(profile.operators)
    operators["qkv"] = H3OperatorProfile(rate=compute("QKV effective FLOP/s", rates["qkv"]))
    operators["out"] = H3OperatorProfile(rate=compute("out effective FLOP/s", rates["out"]))
    profile = replace(profile, operators=operators)
    return (
        shape,
        profile,
        {
            "native_json_sha256": digest(native_path),
            "profile_sqlite_sha256": digest(sql),
            "native_rates": rates,
            "profile_copy_bytes_per_second": copies,
            "profile_vector_elements_per_second": vector_rate,
            "gpu": gpu,
            "scope": "Rate transfer diagnostic. No unprofiled streaming wall time or memory peak was fitted.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape, profile, inputs = calibration(args.directory)
    rows = []
    for path in sorted(args.directory.glob("*.json")):
        raw = read(path)
        if raw["configuration"].get("mode") == "native" or raw["configuration"].get(
            "cuda_profiler_capture"
        ):
            continue
        p = raw["plan"]
        config = H3ExecutionConfig(
            p["q_chunk_tokens"], p["kv_chunk_tokens"], p["qkv_tile_tokens"], p["mlp_tile_tokens"]
        )
        spec = build_h3_block_execution(
            shape,
            config,
            profile,
            callbacks=H3CallbackConfig(
                variant="block25", linear_memory="int8_eager", per_channel_weight_scale=True
            ),
            name=path.stem,
        )
        trace = schedule_execution(spec)
        peak = memory_statistics(trace).pools[0].peak_bytes
        observed_calls = raw["records"][0]["stats"]["mlp_chunks"]
        assert len(spec.metadata["ffn_ranges"]) == observed_calls
        assert spec.metadata["core_workspace_budget_bytes"] == p["estimated_workspace_bytes"]
        row = {
            "source": path.name,
            "sha256": digest(path),
            "projection_tile": p["qkv_tile_tokens"],
            "ffn_tile": p["mlp_tile_tokens"],
            "predicted_seconds": trace.duration_seconds,
            "observed_median_seconds": raw["summary"]["wall_median_seconds"],
            "time_error_percent": 100
            * (trace.duration_seconds / raw["summary"]["wall_median_seconds"] - 1),
            "predicted_activation_peak_bytes": peak,
            "observed_torch_peak_bytes": raw["summary"].get("torch_peak_allocated_bytes"),
            "absolute_memory_error_percent": None,
            "memory_scope": "Predicted activations exclude weights; observed Torch peak includes all live tensors.",
            "predicted_core_workspace_bytes": spec.metadata["core_workspace_budget_bytes"],
            "observed_core_workspace_bytes": p["estimated_workspace_bytes"],
            "ffn_ranges": spec.metadata["ffn_ranges"],
            "observed_ffn_calls": observed_calls,
            "wall_samples": [r["wall_seconds"] for r in raw["records"]],
        }
        rows.append(row)
        print(
            path.name,
            f"predicted {trace.duration_seconds:.6f}s, observed {row['observed_median_seconds']:.6f}s, ",
            f"activation {peak / 2**20:.3f}MiB, FFN calls {observed_calls}",
            flush=True,
        )
        if path.name == "block25_q3840_kv4096_qkv4096_mlp8192.json":
            write_timeline_report(
                trace,
                args.output_dir / "h3_block25_timeline.html",
                json_path=args.output_dir / "h3_block25_timeline.json",
            )
    lower = next(r for r in rows if r["projection_tile"] == 4096 and r["ffn_tile"] == 8192)
    upper = next(r for r in rows if r["projection_tile"] == 4096 and r["ffn_tile"] == 16384)
    observed_growth = upper["observed_torch_peak_bytes"] - lower["observed_torch_peak_bytes"]
    predicted_growth = (
        upper["predicted_activation_peak_bytes"] - lower["predicted_activation_peak_bytes"]
    )
    regular = [r for r in rows if max(r["wall_samples"]) / min(r["wall_samples"]) <= 1.1]
    report = {
        "schema_version": 1,
        "regular_repeat_rule": "max/min <= 1.1, based only on raw repeats; all rows retained",
        "regular_repeat_timing": {
            "count": len(regular),
            "mape_percent": sum(abs(r["time_error_percent"]) for r in regular) / len(regular),
            "max_absolute_percent": max(abs(r["time_error_percent"]) for r in regular),
        },
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p)
            for p in (Path(__file__).resolve().parents[1] / "src/seqattn_core/estimation").rglob(
                "*.py"
            )
        },
        "date": "2026-09-08",
        "calibration": inputs,
        "rows": rows,
        "incremental_memory": {
            "predicted_growth_bytes": predicted_growth,
            "observed_growth_bytes": observed_growth,
            "growth_error_percent": 100 * (predicted_growth / observed_growth - 1),
        },
        "limitations": [
            "Native full-sequence stage rates are not measured small-tile rates; vector and transfer rates use profiling proxies.",
            "Primary latency is the unprofiled wall-time JSON. The unstable configuration-file run is retained.",
            "Eager INT8 workspace is an analytical envelope for the selected implementation, not a measured per-operator sample.",
            "No absolute activation-only memory error is claimed against whole-process Torch peaks.",
        ],
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "incremental_memory": report["incremental_memory"],
                "regular_repeat_timing": report["regular_repeat_timing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
