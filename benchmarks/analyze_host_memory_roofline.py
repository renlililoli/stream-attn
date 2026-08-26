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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze host-memory roofline Q sweeps")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text(encoding="ascii"))
    p_fa4 = prediction["calibration_inputs"]["fa4"]["p_fa4_tflops"]
    b_gbps = prediction["calibration_inputs"]["concurrent_h2d"]["b_concurrent_gbps"]
    q_star = prediction["prediction"]["q_star_predicted"]
    q_95 = prediction["prediction"]["q_95_predicted"]

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    sources = discover_results(args.results)
    for path in sources:
        payload = json.loads(path.read_text(encoding="ascii"))
        if payload.get("status") != "success" or "plan" not in payload:
            continue
        q_tokens = int(payload["plan"]["q_chunk_tokens"])
        grouped[q_tokens].append(payload | {"_path": str(path)})

    rows: list[dict[str, object]] = []
    for q_tokens, payloads in sorted(grouped.items()):
        q_effective_values = [float(item["plan"]["q_effective_tokens"]) for item in payloads]
        q_effective = statistics.median(q_effective_values)
        pipeline_tflops = [float(item["compute_pipeline_effective_tflops"]) for item in payloads]
        wall_tflops = [float(item["effective_tflops"]) for item in payloads]
        predicted_tflops = min(b_gbps * q_effective / 1000.0, p_fa4)
        rows.append(
            {
                "q_tokens": q_tokens,
                "q_effective_tokens": q_effective,
                "q_passes": int(payloads[0]["plan"]["q_passes"]),
                "processes": len(payloads),
                "pipeline_tflops_median": statistics.median(pipeline_tflops),
                "pipeline_tflops_p10": percentile(pipeline_tflops, 0.10),
                "pipeline_tflops_p90": percentile(pipeline_tflops, 0.90),
                "wall_tflops_median": statistics.median(wall_tflops),
                "predicted_tflops": predicted_tflops,
                "observed_over_predicted": statistics.median(pipeline_tflops) / predicted_tflops,
                "observed_over_fa4": statistics.median(pipeline_tflops) / p_fa4,
                "normalized_q": q_effective / q_star,
                "source_paths": [str(item["_path"]) for item in payloads],
            }
        )

    summary = {
        "prediction_path": str(args.prediction),
        "source_paths": [str(path) for path in sources],
        "p_fa4_tflops": p_fa4,
        "b_concurrent_gbps": b_gbps,
        "q_star_predicted": q_star,
        "q_95_predicted": q_95,
        "rows": rows,
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )
    figure, (absolute, normalized) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    q_max = max(8_500, max(int(row["q_tokens"]) for row in rows) + 512)
    q_curve = list(range(1, q_max + 256, 128))
    roof_curve = [min(b_gbps * q / 1000.0, p_fa4) for q in q_curve]
    absolute.plot(q_curve, roof_curve, color="#1f6f5f", linewidth=2.2, label="Predicted roofline")
    absolute.axhline(
        p_fa4, color="#b54708", linestyle="--", linewidth=1.6, label="Resident FA4 roof"
    )
    absolute.axvline(q_star, color="#6b7280", linestyle=":", linewidth=1.4, label="Predicted q*")
    absolute.axvline(
        q_95, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.75, label="Predicted 95% q"
    )
    absolute.scatter(
        [row["q_effective_tokens"] for row in rows],
        [row["pipeline_tflops_median"] for row in rows],
        color="#2563eb",
        edgecolor="white",
        linewidth=0.8,
        s=52,
        zorder=3,
        label="Measured SeqAttn",
    )
    label_offsets = {
        4096: (-34, 7),
        4736: (4, 7),
        5504: (-32, 8),
        5760: (-36, -11),
        5888: (4, 8),
        6272: (4, -11),
        6784: (4, 8),
        8192: (4, -11),
    }
    for row in rows:
        q_tokens = int(row["q_tokens"])
        if q_tokens not in label_offsets:
            continue
        offset = label_offsets[q_tokens]
        absolute.annotate(
            f"q={q_tokens}",
            (row["q_effective_tokens"], row["pipeline_tflops_median"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            va="bottom" if offset[1] > 0 else "top",
        )
    absolute.set_title("Absolute host-memory roofline")
    absolute.set_xlabel("Effective resident Q tokens")
    absolute.set_ylabel("Effective throughput (TFLOP/s)")
    absolute.set_xlim(0, q_max)
    absolute.set_ylim(0, p_fa4 * 1.12)
    absolute.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    absolute.legend(loc="lower right")

    x_curve = [index / 200 for index in range(401)]
    normalized.plot(
        x_curve,
        [min(x, 1.0) for x in x_curve],
        color="#1f6f5f",
        linewidth=2.2,
        label="Ideal y=min(x,1)",
    )
    normalized.axvline(1.0, color="#6b7280", linestyle=":", linewidth=1.4)
    normalized.scatter(
        [row["normalized_q"] for row in rows],
        [row["observed_over_fa4"] for row in rows],
        color="#7c3aed",
        edgecolor="white",
        linewidth=0.8,
        s=52,
        zorder=3,
        label="Measured SeqAttn",
    )
    for row in rows:
        q_tokens = int(row["q_tokens"])
        if q_tokens not in label_offsets:
            continue
        offset = label_offsets[q_tokens]
        normalized.annotate(
            f"{q_tokens}",
            (row["normalized_q"], row["observed_over_fa4"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            va="bottom" if offset[1] > 0 else "top",
        )
    normalized.set_title("Normalized model comparison")
    normalized.set_xlabel("q_effective / q*_predicted")
    normalized.set_ylabel("P_SeqAttn / P_FA4")
    normalized.set_xlim(0, 1.6)
    normalized.set_ylim(0, 1.08)
    normalized.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    normalized.legend(loc="lower right")

    figure.suptitle("RTX 5090, N=524,288, BF16 MHA, fixed K/V chunk=4,096", fontsize=13)
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
