from __future__ import annotations

import argparse
import json
import math
import statistics
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the A30 host-memory roofline sweep")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--partial-label")
    args = parser.parse_args()

    prediction = json.loads(args.prediction.read_text(encoding="ascii"))
    sweep = json.loads(args.sweep.read_text(encoding="ascii"))
    p_fa2 = prediction["calibration_inputs"]["resident_fa2"]["p_fa2_tflops"]
    b_gbps = prediction["calibration_inputs"]["concurrent_h2d"][
        "b_concurrent_gbps"
    ]
    q_star = prediction["prediction"]["q_star_predicted"]
    q_95 = prediction["prediction"]["q_95_predicted"]

    rows: list[dict[str, object]] = []
    for result in sweep["results"]:
        if result.get("status") != "success":
            continue
        q_tokens = int(result["q_chunk_tokens"])
        q_effective = float(result["q_effective_tokens"])
        pipeline_seconds = [float(value) for value in result["pipeline_seconds"]]
        flop = 4 * 56 * 128 * 409600 * 409600
        pipeline_tflops = [flop / seconds / 1e12 for seconds in pipeline_seconds]
        measured = statistics.median(pipeline_tflops)
        predicted = min(b_gbps * q_effective / 1000.0, p_fa2)
        rows.append(
            {
                "q_tokens": q_tokens,
                "q_effective_tokens": q_effective,
                "q_passes": int(result["q_passes"]),
                "pipeline_tflops_median": measured,
                "pipeline_tflops_p10": percentile(pipeline_tflops, 0.10),
                "pipeline_tflops_p90": percentile(pipeline_tflops, 0.90),
                "predicted_tflops": predicted,
                "observed_over_predicted": measured / predicted,
                "observed_over_fa2": measured / p_fa2,
                "normalized_q": q_effective / q_star,
            }
        )
    rows.sort(key=lambda row: int(row["q_tokens"]))

    summary = {
        "status": "partial" if sweep.get("status") != "success" else "complete",
        "partial_label": args.partial_label,
        "prediction_path": str(args.prediction),
        "sweep_path": str(args.sweep),
        "completed_points": len(rows),
        "requested_points": len(sweep["configuration"]["q_values"]),
        "p_fa2_tflops": p_fa2,
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
    q_max = max([10_000, *(int(row["q_tokens"]) for row in rows)])
    q_curve = list(range(1, q_max + 256, 128))
    roof_curve = [min(b_gbps * q / 1000.0, p_fa2) for q in q_curve]
    absolute.plot(
        q_curve,
        roof_curve,
        color="#1f6f5f",
        linewidth=2.2,
        label="Predicted roofline",
    )
    absolute.axhline(
        p_fa2,
        color="#b54708",
        linestyle="--",
        linewidth=1.6,
        label="Resident FA2 roof",
    )
    absolute.axvline(
        q_star,
        color="#6b7280",
        linestyle=":",
        linewidth=1.4,
        label="Predicted q*",
    )
    absolute.axvline(
        q_95,
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="Predicted 95% q",
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
    absolute_label_positions = {
        7424: (-6, 10, "right"),
        7680: (0, -16, "center"),
        8192: (8, 10, "left"),
        12288: (-6, 8, "right"),
    }
    for row in rows:
        offset_x, offset_y, alignment = absolute_label_positions.get(
            int(row["q_tokens"]), (4, 6, "left")
        )
        absolute.annotate(
            f"q={row['q_tokens']}",
            (row["q_effective_tokens"], row["pipeline_tflops_median"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            horizontalalignment=alignment,
        )
    absolute.set_title("Absolute host-memory roofline")
    absolute.set_xlabel("Effective resident Q tokens")
    absolute.set_ylabel("Effective throughput (TFLOP/s)")
    absolute.set_xlim(0, q_max)
    absolute.set_ylim(0, p_fa2 * 1.12)
    absolute.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    absolute.legend(loc="lower right")

    x_curve = [index / 200 for index in range(0, 401)]
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
        [row["observed_over_fa2"] for row in rows],
        color="#7c3aed",
        edgecolor="white",
        linewidth=0.8,
        s=52,
        zorder=3,
        label="Measured SeqAttn",
    )
    normalized_label_positions = {
        7424: (-6, 10, "right"),
        7680: (0, -16, "center"),
        8192: (8, 10, "left"),
        12288: (-6, 8, "right"),
    }
    for row in rows:
        offset_x, offset_y, alignment = normalized_label_positions.get(
            int(row["q_tokens"]), (4, 6, "left")
        )
        normalized.annotate(
            str(row["q_tokens"]),
            (row["normalized_q"], row["observed_over_fa2"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            horizontalalignment=alignment,
        )
    normalized.set_title("Normalized model comparison")
    normalized.set_xlabel("q_effective / q*_predicted")
    normalized.set_ylabel("P_SeqAttn / P_FA2")
    normalized.set_xlim(0, 1.8)
    normalized.set_ylim(0, 1.08)
    normalized.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    normalized.legend(loc="lower right")

    title = "A30, N=409,600, BF16 MHA, fixed K/V chunk=4,096"
    if args.partial_label:
        title = f"{title} ({args.partial_label})"
    figure.suptitle(title, fontsize=13)
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
