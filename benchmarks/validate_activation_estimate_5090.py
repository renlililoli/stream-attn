"""Validate offline estimates against retained RTX 5090 measurements; no GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from seqattn_core.estimation import (
    BufferSpec,
    ExecutionSpec,
    MemoryPool,
    OperationSpec,
    RateProfile,
    memory_statistics,
    schedule_execution,
    write_timeline_report,
)

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = (
    ROOT
    / "docs/experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824/comparison_observations.json"
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_buffers(config, q, kv, uses=None):
    uses = {} if uses is None else uses
    q_width = config["q_heads"] * config["head_dim"]
    kv_width = config["kv_heads"] * config["head_dim"]
    sizes = {
        "Q": (q * q_width * 2, "Q"),
        "accumulator": (q * q_width * 4, "FP32 state"),
        "running_max": (q * config["q_heads"] * 4, "FP32 state"),
        "running_sum": (q * config["q_heads"] * 4, "FP32 state"),
        "output": (q * q_width * 2, "output buffer"),
    }
    for slot in range(2):
        sizes[f"K.{slot}"] = (kv * kv_width * 2, "K")
        sizes[f"V.{slot}"] = (kv * kv_width * 2, "V")
    return tuple(
        BufferSpec(name, "GPU.HBM", size, component, tuple(uses.get(name, ())), persistent=True)
        for name, (size, component) in sizes.items()
    )


def attention_execution(config, q, kv, p_flops, bandwidth, name):
    """Actual resident-Q/2-slot-KV dependency pattern, with frozen effective rates.

    No D2H/launch/finalize rate was independently calibrated for both memory
    policies. These terms are deliberately omitted, not fitted to the results.
    The measured compute_pipeline includes finalization and possible output waits.
    """
    attention = RateProfile(
        "resident FA4 roof",
        p_flops,
        ("compute",),
        provenance="independent pre-sweep resident FA4 CUDA-event median",
    )
    transfer = RateProfile(
        "concurrent H2D",
        bandwidth,
        ("H2D",),
        kind="io",
        provenance="independent pre-sweep concurrent pinned H2D calibration",
    )
    operations, uses = [], {}

    def add(name, profile, work, deps, buffers, component):
        operations.append(
            OperationSpec(
                name,
                profile.seconds(work),
                profile.resources,
                tuple(dict.fromkeys(x for x in deps if x)),
                profile.kind,
                component,
                profile.provenance,
            )
        )
        for buffer in buffers:
            uses.setdefault(buffer, []).append(name)
        return name

    length = config["tokens"]
    q_width = config["q_heads"] * config["head_dim"]
    kv_width = config["kv_heads"] * config["head_dim"]
    last_q, slots = None, [None, None]
    for q_start in range(0, length, q):
        q_tokens = min(q, length - q_start)
        prefix = f"q{q_start}"
        q_ready = add(f"{prefix}.copy", transfer, q_tokens * q_width * 2, (last_q,), ("Q",), "Q")
        ready = q_ready
        for index, k_start in enumerate(range(0, length, kv)):
            k_tokens = min(kv, length - k_start)
            slot = index % 2
            buffers = (f"K.{slot}", f"V.{slot}")
            loaded = add(
                f"{prefix}.kv{k_start}.copy",
                transfer,
                2 * k_tokens * kv_width * 2,
                (q_ready, slots[slot]),
                buffers,
                "KV transfer",
            )
            ready = add(
                f"{prefix}.kv{k_start}.attention",
                attention,
                4 * q_tokens * k_tokens * q_width,
                (ready, loaded),
                ("Q", "accumulator", "running_max", "running_sum", *buffers),
                "attention",
            )
            slots[slot] = ready
        last_q = ready
    return ExecutionSpec(
        name,
        (MemoryPool("GPU.HBM"),),
        tuple(operations),
        tensor_buffers(config, q, kv, uses),
        metadata={
            "q_chunk_tokens": q,
            "kv_tile_tokens": kv,
            "tokens": length,
            "q_heads": config["q_heads"],
            "kv_heads": config["kv_heads"],
            "head_dim": config["head_dim"],
            "attention_flops_per_second": p_flops,
            "h2d_bytes_per_second": bandwidth,
            "attention_backend": "Triton",
            "compute_calibration_backend": "resident FA4",
        },
        assumptions=(
            "Frozen independent FA4 throughput and concurrent H2D bandwidth, no fitted constants.",
            "BF16 MHA, non-causal single segment, two KV slots and one host-output slot.",
            "Core H2D/attention estimate omits finalize, CPU launch overhead and output-copy waits.",
            "Allocated tensor storage excludes the unallocated 32 MiB workspace reserve.",
            "Reported memory measurements are allocator peaks, not per-buffer lifetime samples.",
        ),
    )


def summary(rows, key):
    values = [row[key] for row in rows]
    return {
        "count": len(values),
        "mean_absolute_percent": statistics.mean(map(abs, values)),
        "mean_signed_percent": statistics.mean(values),
        "min_signed_percent": min(values),
        "max_signed_percent": max(values),
        "max_absolute_percent": max(map(abs, values)),
    }


def validate_series(label, observations, evidence, output):
    prediction_path = ROOT / observations["prediction"]["path"]
    prediction = read(prediction_path)
    config = prediction["configuration"]
    rates = prediction["calibration_inputs"]
    p_flops = rates["fa4"]["p_fa4_tflops"] * 1e12
    h2d = rates.get("concurrent_h2d", rates.get("interleaved_h2d"))
    bandwidth = h2d["b_concurrent_bytes_per_second"]
    calibration_integrity = []
    for calibration in (rates["fa4"], h2d):
        path = evidence / calibration["path"].split("/workspace/benchmarks/results/")[-1]
        actual_hash = digest(path)
        if actual_hash != calibration["sha256"]:
            raise ValueError(f"calibration hash mismatch: {path}")
        calibration_integrity.append(
            {"path": str(path.relative_to(evidence)), "sha256": actual_hash}
        )
    rows = []
    for observation in observations["rows"]:
        raw_path = (
            evidence / observation["source_paths"][0].split("/workspace/benchmarks/results/")[-1]
        )
        raw = read(raw_path)
        if raw["status"] != "success":
            raise ValueError(f"curated observation is not successful: {raw_path}")
        for key in ("tokens", "q_heads", "kv_heads", "head_dim", "dtype", "causal"):
            if raw["configuration"][key] != config[key]:
                raise ValueError(f"configuration mismatch for {key}: {raw_path}")
        plan = raw["plan"]
        if plan["num_kv_buffers"] != 2 or plan["num_output_buffers"] != 1:
            raise ValueError("this comparison adapter requires two KV and one output buffer")
        q, kv = plan["q_chunk_tokens"], plan["kv_chunk_tokens"]
        if kv != config["kv_chunk_tokens"]:
            raise ValueError(f"KV calibration mismatch: {raw_path}")
        spec = attention_execution(config, q, kv, p_flops, bandwidth, f"{label}, Q={q}")
        trace = schedule_execution(spec)
        memory = memory_statistics(trace).pools[0]
        measured_seconds = raw["mean_compute_pipeline_seconds"]
        measured_bytes = round(raw["torch_peak_allocated_mib"] * 2**20)
        flops = 4 * config["tokens"] ** 2 * config["q_heads"] * config["head_dim"]
        row = {
            "q_tokens": q,
            "kv_tokens": kv,
            "q_passes": plan["q_passes"],
            "predicted_seconds": trace.duration_seconds,
            "measured_seconds": measured_seconds,
            "time_error_percent": 100 * (trace.duration_seconds / measured_seconds - 1),
            "predicted_tflops": flops / trace.duration_seconds / 1e12,
            "measured_tflops": raw["compute_pipeline_effective_tflops"],
            "frozen_roofline_seconds": flops / observation["predicted_tflops"] / 1e12,
            "predicted_allocated_bytes": memory.peak_bytes,
            "measured_allocated_bytes": measured_bytes,
            "memory_error_percent": 100 * (memory.peak_bytes / measured_bytes - 1),
            "allocation_residual_bytes": measured_bytes - memory.peak_bytes,
            "predicted_workspace_with_reserve_bytes": memory.peak_bytes + 32 * 2**20,
            "recorded_workspace_bytes": round(plan["estimated_workspace_mib"] * 2**20),
            "memory_components": memory.components_at_peak,
            "source": str(raw_path.relative_to(evidence)),
            "source_sha256": digest(raw_path),
            "repeats": raw["configuration"]["repeats"],
            "environment": raw["environment"],
            "memory_probe_skipped": raw["memory_probe_skipped"],
        }
        if row["predicted_workspace_with_reserve_bytes"] != row["recorded_workspace_bytes"]:
            raise ValueError(f"workspace accounting mismatch: {raw_path}")
        rows.append(row)
        print(
            label,
            q,
            f"time {row['time_error_percent']:+.3f}%, memory {row['memory_error_percent']:+.3f}%",
            flush=True,
        )
        # One complete estimate per topology is enough to inspect the actual event schedule.
        if q == (4096 if label == "membind5" else 3840):
            write_timeline_report(trace, output / f"{label}_timeline.html")
        (output / f"{label}_partial.json").write_text(json.dumps(rows, indent=2) + "\n")
    predicted_threshold = 0.95 * max(r["predicted_tflops"] for r in rows)
    measured_threshold = 0.95 * max(r["measured_tflops"] for r in rows)
    minimum = lambda metric, threshold: min(
        (r for r in rows if r[metric] >= threshold), key=lambda r: r["q_tokens"]
    )
    return {
        "name": label,
        "prediction_source": str(prediction_path.relative_to(ROOT)),
        "calibration_frozen_before_evaluation": prediction["prediction_created_before_q_sweep"],
        "calibration_git_commit": prediction["calibration_git_commit"],
        "calibration": calibration_integrity,
        "configuration": config,
        "attention_tflops": p_flops / 1e12,
        "h2d_gbps": bandwidth / 1e9,
        "excluded_q_from_original_study": observations["excluded_q"],
        "rows": rows,
        "time_error": summary(rows, "time_error_percent"),
        "memory_error": summary(rows, "memory_error_percent"),
        "minimum_q_at_common_fa4_target": {
            "target_tflops": 0.95 * p_flops / 1e12,
            "predicted": minimum("predicted_tflops", 0.95 * p_flops / 1e12)["q_tokens"],
            "observed": minimum("measured_tflops", 0.95 * p_flops / 1e12)["q_tokens"],
        },
        "minimum_q_at_95_percent_of_each_plateau": {
            "predicted": minimum("predicted_tflops", predicted_threshold)["q_tokens"],
            "observed": minimum("measured_tflops", measured_threshold)["q_tokens"],
            "predicted_threshold_tflops": predicted_threshold,
            "observed_threshold_tflops": measured_threshold,
        },
    }


def memory_sweep(evidence):
    rows = []
    for path in sorted((evidence / "rtx5090_dram_workspace_524k_low_20260824").glob("*.json")):
        raw = read(path)
        for measured in raw["rows"]:
            if measured["status"] != "success":
                rows.append(
                    {"status": measured["status"], "source": str(path.relative_to(evidence))}
                )
                continue
            q, kv = measured["q_chunk_tokens"], measured["kv_chunk_tokens"]
            buffers = tensor_buffers(raw["configuration"], q, kv)
            predicted = sum(buffer.size_bytes for buffer in buffers)
            if predicted + 32 * 2**20 != measured["estimated_workspace_bytes"]:
                raise ValueError(f"buffer policy not accounted for: {path}")
            actual = measured["torch_peak_allocated_bytes"]
            rows.append(
                {
                    "source": str(path.relative_to(evidence)),
                    "source_sha256": digest(path),
                    "status": "success",
                    "q_tokens": q,
                    "kv_tokens": kv,
                    "predicted_allocated_bytes": predicted,
                    "measured_allocated_bytes": actual,
                    "memory_error_percent": 100 * (predicted / actual - 1),
                    "nvml_process_peak_bytes": measured["nvml_process_peak_bytes"],
                    "time_validation": "excluded: no matching KV=2048 bandwidth calibration",
                }
            )
    return rows


def h3_scope_audit():
    rows = []
    for path in sorted((ROOT / "benchmarks/results/h3_qkv_recompute_20260827").glob("*.json")):
        raw = read(path)
        c, result = raw["configuration"], raw["summary"]
        projection = raw["records"][0]["stats"]["projection"]
        q_passes = (c["tokens"] + c["q_chunk_tokens"] - 1) // c["q_chunk_tokens"]
        kv_tiles = (c["tokens"] + c["kv_chunk_tokens"] - 1) // c["kv_chunk_tokens"]
        generic_host_bytes = (
            2 * raw["host_activation"]["hidden_bytes_each"]
            + raw["host_activation"]["materialized_qkv_bytes"]
        )
        recorded_host_bytes = result["logical_host_activation_bytes"]
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "source_sha256": digest(path),
                "mode": c["mode"],
                "tokens": c["tokens"],
                "q_tokens": c["q_chunk_tokens"],
                "wall_median_seconds": result["wall_median_seconds"],
                "recorded_core_workspace_bytes": raw["plan"]["estimated_workspace_bytes"],
                "torch_allocated_peak_bytes": result["torch_peak_allocated_bytes"],
                "nvml_process_peak_bytes": round(result["nvml_process_peak_mib"] * 2**20),
                "expected_direct_write_q_calls": q_passes if c["mode"] == "recompute" else None,
                "observed_q_projection_calls": projection["recompute_q_projection_chunks"],
                "expected_direct_write_kv_calls": q_passes * kv_tiles
                if c["mode"] == "recompute"
                else None,
                "observed_kv_projection_calls": projection["recompute_kv_projection_chunks"],
                "generic_template_logical_host_bytes": generic_host_bytes,
                "recorded_logical_host_bytes": recorded_host_bytes,
                "logical_host_difference_percent": 100
                * (generic_host_bytes / recorded_host_bytes - 1),
                "logical_host_difference_scope": "distinct input/output template versus in-place materialized H3; not RSS",
                "valid_current_block_comparison": False,
                "reason": (
                    "Legacy recompute projection subtiles differ from current direct-write mode; "
                    "no isolated activation-only peak or matched GEMM calibration."
                    if c["mode"] == "recompute"
                    else "Core workspace and whole Torch/NVML peaks have different ownership; "
                    "no isolated activation-only peak or matched GEMM calibration."
                ),
            }
        )
    return rows


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), constrained_layout=True)
    for column, series in enumerate(result["series"]):
        rows = series["rows"]
        q = [r["q_tokens"] for r in rows]
        ax = axes[0, column]
        ax.plot(
            q,
            [r["measured_seconds"] for r in rows],
            "o-",
            label="Measured compute pipeline",
            color="#1d7278",
        )
        ax.plot(
            q,
            [r["predicted_seconds"] for r in rows],
            "s--",
            label="Event model, frozen rates",
            color="#bd713d",
        )
        ax.set(
            title=f"{series['name']} · {series['h2d_gbps']:.3f} GB/s",
            xlabel="Resident Q tokens",
            ylabel="Seconds",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        ax = axes[1, column]
        ax.plot(
            q,
            [r["measured_allocated_bytes"] / 2**20 for r in rows],
            "o-",
            label="Torch allocated peak",
            color="#1d7278",
        )
        ax.plot(
            q,
            [r["predicted_allocated_bytes"] / 2**20 for r in rows],
            "s--",
            label="Physical tensor estimate",
            color="#bd713d",
        )
        ax.set(
            xlabel="Resident Q tokens",
            ylabel="MiB",
            title=f"Memory MAPE {series['memory_error']['mean_absolute_percent']:.3f}%",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(
        "RTX 5090 · BF16 MHA · 524,288 tokens · 56 heads × 128 · KV 4,096\nIndependent FA4/H2D calibration; actual attention backend: Triton",
        fontsize=13,
    )
    fig.savefig(output / "comparison.svg")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = read(COMPARISON)
    series = [
        validate_series(name, observations[key], args.evidence_root, args.output_dir)
        for name, key in (("membind5", "baseline"), ("interleave57", "interleaved"))
    ]
    sweep = memory_sweep(args.evidence_root)
    timing_rows = [r for s in series for r in s["rows"]]
    memory_rows = timing_rows + [r for r in sweep if r["status"] == "success"]
    result = {
        "schema_version": 1,
        "source_roots": {
            "repository": ".",
            "external_benchmark_results": "../../workspace/benchmarks/results",
        },
        "validation_date": "2026-09-08",
        "validation_script_sha256": digest(Path(__file__)),
        "series": series,
        "all_timing_error": summary(timing_rows, "time_error_percent"),
        "all_memory_error": summary(memory_rows, "memory_error_percent"),
        "memory_only_kv2048_sweep": sweep,
        "h3_scope_audit": h3_scope_audit(),
        "estimator_source_sha256": {
            p.name: digest(p) for p in (ROOT / "src/seqattn_core/estimation").glob("*.py")
        },
        "limitations": [
            "No rates were fitted to evaluation points. FA4 calibration and Triton execution differ.",
            "Core H2D/attention timing omits finalize, output waits and launch overhead.",
            "Allocator peaks validate tensor sizes, not time-resolved lifetimes or whole-process NVML.",
            "KV=2048 sweep validates memory only; its timing topology was not mixed into KV=4096.",
            "No matched independent GEMM and activation-only H3 measurement available.",
        ],
    }
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output_dir)
    print(
        json.dumps(
            {
                "timing": result["all_timing_error"],
                "memory": result["all_memory_error"],
                "minimum_q": {
                    s["name"]: s["minimum_q_at_95_percent_of_each_plateau"] for s in series
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
