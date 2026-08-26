from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def discover_results(paths: list[Path]) -> list[Path]:
    results: list[Path] = []
    for path in paths:
        if path.is_dir():
            results.extend(sorted(path.rglob("q_*.json")))
        elif path.is_file():
            results.append(path)
    return sorted(set(results))


def prediction_inputs(path: Path) -> dict[str, float | str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    calibration = payload["calibration_inputs"]
    h2d = calibration.get("concurrent_h2d") or calibration.get("interleaved_h2d")
    if h2d is None:
        raise ValueError(f"no H2D calibration found in {path}")
    return {
        "path": str(path),
        "p_fa4_tflops": float(calibration["fa4"]["p_fa4_tflops"]),
        "b_gbps": float(h2d["b_concurrent_gbps"]),
        "q_star": float(payload["prediction"]["q_star_predicted"]),
        "q_95": float(payload["prediction"]["q_95_predicted"]),
    }


def summarize_results(
    paths: list[Path],
    *,
    prediction: dict[str, float | str],
    excluded_q: set[int],
) -> tuple[list[dict[str, object]], list[str]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    sources = discover_results(paths)
    for path in sources:
        payload = json.loads(path.read_text(encoding="ascii"))
        if payload.get("status") != "success" or "plan" not in payload:
            continue
        q_tokens = int(payload["plan"]["q_chunk_tokens"])
        if q_tokens in excluded_q:
            continue
        grouped[q_tokens].append(payload | {"_path": str(path)})

    b_gbps = float(prediction["b_gbps"])
    p_fa4 = float(prediction["p_fa4_tflops"])
    q_star = float(prediction["q_star"])
    rows: list[dict[str, object]] = []
    for q_tokens, payloads in sorted(grouped.items()):
        q_effective = statistics.median(
            float(item["plan"]["q_effective_tokens"]) for item in payloads
        )
        pipeline = [float(item["compute_pipeline_effective_tflops"]) for item in payloads]
        predicted = min(b_gbps * q_effective / 1000.0, p_fa4)
        rows.append(
            {
                "q_tokens": q_tokens,
                "q_effective_tokens": q_effective,
                "q_passes": int(payloads[0]["plan"]["q_passes"]),
                "processes": len(payloads),
                "pipeline_tflops_median": statistics.median(pipeline),
                "pipeline_tflops_p10": percentile(pipeline, 0.10),
                "pipeline_tflops_p90": percentile(pipeline, 0.90),
                "predicted_tflops": predicted,
                "observed_over_predicted": statistics.median(pipeline) / predicted,
                "observed_over_fa4": statistics.median(pipeline) / p_fa4,
                "normalized_q": q_effective / q_star,
                "source_paths": [str(item["_path"]) for item in payloads],
            }
        )
    return rows, [str(path) for path in sources]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two host-memory roofline Q sweeps")
    parser.add_argument("--baseline-prediction", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, nargs="+", required=True)
    parser.add_argument("--interleaved-prediction", type=Path, required=True)
    parser.add_argument("--interleaved-results", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-baseline-q", type=int, nargs="*", default=())
    parser.add_argument("--exclude-interleaved-q", type=int, nargs="*", default=())
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    baseline_prediction = prediction_inputs(args.baseline_prediction)
    interleaved_prediction = prediction_inputs(args.interleaved_prediction)
    baseline_rows, baseline_sources = summarize_results(
        args.baseline_results,
        prediction=baseline_prediction,
        excluded_q=set(args.exclude_baseline_q),
    )
    interleaved_rows, interleaved_sources = summarize_results(
        args.interleaved_results,
        prediction=interleaved_prediction,
        excluded_q=set(args.exclude_interleaved_q),
    )
    if not baseline_rows or not interleaved_rows:
        raise ValueError("both sweeps must contain at least one successful result")

    summary = {
        "baseline": {
            "prediction": baseline_prediction,
            "excluded_q": sorted(args.exclude_baseline_q),
            "source_paths": baseline_sources,
            "rows": baseline_rows,
        },
        "interleaved": {
            "prediction": interleaved_prediction,
            "excluded_q": sorted(args.exclude_interleaved_q),
            "source_paths": interleaved_sources,
            "rows": interleaved_rows,
        },
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "svg.fonttype": "none",
        }
    )
    figure, (absolute, normalized) = plt.subplots(1, 2, figsize=(12.2, 4.4))
    p_fa4 = float(baseline_prediction["p_fa4_tflops"])
    q_max = max(
        8500,
        *(int(row["q_tokens"]) for row in baseline_rows),
        *(int(row["q_tokens"]) for row in interleaved_rows),
    )
    q_curve = list(range(1, q_max + 128, 64))

    series = (
        (
            "Single NUMA node",
            baseline_prediction,
            baseline_rows,
            "#2563eb",
            "o",
        ),
        (
            "Interleave node5+7",
            interleaved_prediction,
            interleaved_rows,
            "#ea580c",
            "s",
        ),
    )
    for label, prediction, rows, color, marker in series:
        b_gbps = float(prediction["b_gbps"])
        q_star = float(prediction["q_star"])
        roof = [min(b_gbps * q / 1000.0, p_fa4) for q in q_curve]
        absolute.plot(
            q_curve,
            roof,
            color=color,
            linewidth=2.1,
            label=f"{label} roof ({b_gbps:.2f} GB/s)",
        )
        absolute.axvline(q_star, color=color, linestyle=":", linewidth=1.3)
        absolute.scatter(
            [row["q_effective_tokens"] for row in rows],
            [row["pipeline_tflops_median"] for row in rows],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            s=50,
            zorder=3,
            label=f"{label} measured",
        )
        normalized.scatter(
            [row["normalized_q"] for row in rows],
            [row["observed_over_fa4"] for row in rows],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            s=50,
            zorder=3,
            label=label,
        )

    absolute.axhline(
        p_fa4,
        color="#475569",
        linestyle="--",
        linewidth=1.5,
        label="Resident FA4 roof",
    )
    absolute.set_title("Bandwidth change shifts the Q knee")
    absolute.set_xlabel("Effective resident Q tokens")
    absolute.set_ylabel("Effective throughput (TFLOP/s)")
    absolute.set_xlim(2500, q_max)
    absolute.set_ylim(120, p_fa4 * 1.08)
    absolute.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    absolute.legend(loc="lower right")

    x_curve = [index / 200 for index in range(361)]
    normalized.plot(
        x_curve,
        [min(x, 1.0) for x in x_curve],
        color="#166534",
        linewidth=2.1,
        label="Ideal y=min(x,1)",
    )
    normalized.axvline(1.0, color="#475569", linestyle=":", linewidth=1.3)
    normalized.set_title("Machine-balance normalization")
    normalized.set_xlabel("q_effective / q*_predicted")
    normalized.set_ylabel("P_SeqAttn / P_FA4")
    normalized.set_xlim(0.5, 1.6)
    normalized.set_ylim(0.55, 1.05)
    normalized.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    normalized.legend(loc="lower right")

    figure.suptitle(
        "RTX 5090, N=524,288, BF16 MHA, fixed K/V chunk=4,096",
        fontsize=13,
    )
    figure.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_figure, bbox_inches="tight")
    if args.output_figure.suffix.lower() == ".svg":
        svg = args.output_figure.read_text(encoding="utf-8")
        args.output_figure.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
