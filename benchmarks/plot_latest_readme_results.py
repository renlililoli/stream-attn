from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
A30_OBSERVATIONS = (
    ROOT / "docs/experiments/a30_host_memory_roofline_experiment0_20260824/observations.json"
)
RTX_COMPARISON = (
    ROOT
    / "docs/experiments/rtx5090_host_memory_roofline_experiment0b_interleave57_20260824"
    / "comparison_observations.json"
)
INK = "#172033"
MUTED = "#64748B"
GRID = "#D7DEE8"
MODEL = "#475569"
TEAL = "#0F766E"
ORANGE = "#C2410C"
GOLD = "#A16207"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "seqattn-readme-20260824",
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


def save_svg(fig: plt.Figure, output: Path, *, date: str = "2026-08-24") -> None:
    fig.savefig(output, format="svg", dpi=144, metadata={"Date": date})
    lines = output.read_text().splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def plot_a30(output: Path) -> None:
    data = load_json(A30_OBSERVATIONS)
    rows = data["rows"]
    bandwidth = data["b_concurrent_gbps"]
    resident_roof = data["p_fa2_tflops"]
    predicted_knee = data["q_star_predicted"]
    observed_knee = data["observed_knee"]["intersection_estimate"]
    plateau = data["plateau"]["pipeline_tflops_median"]

    x = [row["q_effective_tokens"] for row in rows]
    y = [row["pipeline_tflops_median"] for row in rows]
    yerr = [
        [value - row["pipeline_tflops_p10"] for value, row in zip(y, rows)],
        [row["pipeline_tflops_p90"] - value for value, row in zip(y, rows)],
    ]
    model_x = [1800 + index * 40 for index in range(381)]
    model_y = [min(bandwidth * value / 1000.0, resident_roof) for value in model_x]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.plot(
        model_x,
        model_y,
        color=MODEL,
        linewidth=2.1,
        linestyle=(0, (6, 4)),
        label="Frozen host-memory roofline",
    )
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=TEAL,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=1.8,
        capsize=2.5,
        label="Measured FA2 streaming",
        zorder=3,
    )
    ax.axhline(
        resident_roof,
        color=GOLD,
        linewidth=1.4,
        linestyle=":",
        label=f"Resident FA2 roof: {resident_roof:.2f} TFLOP/s",
    )
    ax.axhline(
        plateau,
        color=TEAL,
        linewidth=1.2,
        linestyle=(0, (2, 3)),
        alpha=0.8,
        label=f"Streaming plateau: {plateau:.2f} TFLOP/s",
    )
    ax.axvline(predicted_knee, color=MODEL, linewidth=1.2, linestyle="--")
    ax.axvline(observed_knee, color=ORANGE, linewidth=1.5)
    ax.annotate(
        f"predicted q*  {predicted_knee:,.0f}",
        xy=(predicted_knee, 19),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=MODEL,
        fontsize=9,
    )
    ax.annotate(
        f"inferred q*  {observed_knee:,.0f}  (-2.01%)",
        xy=(observed_knee, 13),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=ORANGE,
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title(
        "A30: measured FA2 streaming follows the frozen host-memory roofline",
        loc="left",
    )
    ax.set_xlabel("Effective resident-Q tokens")
    ax.set_ylabel("Pipeline throughput (TFLOP/s)")
    ax.set_xlim(1700, 16800)
    ax.set_ylim(0, 100)
    finish_axes(ax)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.text(
        0.075,
        0.015,
        "409,600 tokens | BF16 MHA 56x128 | K/V chunk 4,096 | "
        "1 warmup + 3 measured runs per Q | 2026-08-24",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.055, 0.99, 0.98))
    save_svg(fig, output)
    plt.close(fig)


def row_by_q(rows: list[dict], q_tokens: int) -> dict:
    return next(row for row in rows if row["q_tokens"] == q_tokens)


def plot_rtx5090(output: Path) -> None:
    data = load_json(RTX_COMPARISON)
    baseline = data["baseline"]
    interleaved = data["interleaved"]
    resident_roof = baseline["prediction"]["p_fa4_tflops"]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    model_x = [0.68 + index * 0.002 for index in range(402)]
    model_y = [min(resident_roof * value, resident_roof) for value in model_x]
    ax.plot(
        model_x,
        model_y,
        color=MODEL,
        linewidth=2.1,
        linestyle=(0, (6, 4)),
        label="Frozen normalized roofline",
    )
    ax.axvline(1.0, color=MODEL, linewidth=1.2, linestyle="--")
    ax.axhline(
        resident_roof,
        color=GOLD,
        linewidth=1.4,
        linestyle=":",
        label=f"Resident FA4 roof: {resident_roof:.2f} TFLOP/s",
    )

    series = [
        (baseline, TEAL, "o", "node5 only | 37.28 GB/s"),
        (interleaved, ORANGE, "s", "interleave node5+7 | 56.72 GB/s"),
    ]
    highlighted = {"baseline": {5888, 6272, 6784}, "interleaved": {3840, 4096, 4480}}
    for policy, color, marker, label in series:
        rows = policy["rows"]
        x = [row["normalized_q"] for row in rows]
        y = [row["pipeline_tflops_median"] for row in rows]
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(
            x,
            y,
            s=36,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            linewidth=1.6,
            label=label,
            zorder=3,
        )
        key = "baseline" if policy is baseline else "interleaved"
        selected = [row for row in rows if row["q_tokens"] in highlighted[key]]
        ax.scatter(
            [row["normalized_q"] for row in selected],
            [row["pipeline_tflops_median"] for row in selected],
            s=72,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )

    for baseline_q, interleaved_q in [(5888, 3840), (6272, 4096), (6784, 4480)]:
        left = row_by_q(baseline["rows"], baseline_q)
        right = row_by_q(interleaved["rows"], interleaved_q)
        ax.plot(
            [left["normalized_q"], right["normalized_q"]],
            [left["pipeline_tflops_median"], right["pipeline_tflops_median"]],
            color="#A8B3C2",
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )

    single_corrected = 5748.772 / baseline["prediction"]["q_star"]
    interleaved_corrected = 3754.300 / interleaved["prediction"]["q_star"]
    ax.axvline(single_corrected, color=TEAL, linewidth=1.5, alpha=0.9)
    ax.axvline(interleaved_corrected, color=ORANGE, linewidth=1.5, alpha=0.9)
    ax.annotate(
        "node5 inferred knee  +0.48%",
        xy=(single_corrected, 146),
        xytext=(10, 0),
        textcoords="offset points",
        color=TEAL,
        fontsize=9,
        fontweight="bold",
        ha="left",
    )
    ax.annotate(
        "interleaved inferred knee  -0.18%",
        xy=(interleaved_corrected, 140),
        xytext=(-10, 0),
        textcoords="offset points",
        color=ORANGE,
        fontsize=9,
        fontweight="bold",
        ha="right",
    )
    ax.set_title(
        "RTX 5090: measured bandwidth shifts the knee as predicted",
        loc="left",
    )
    ax.set_xlabel("Effective resident Q / independently predicted q*")
    ax.set_ylabel("Pipeline throughput (TFLOP/s)")
    ax.set_xlim(0.68, 1.47)
    ax.set_ylim(135, 220)
    finish_axes(ax)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.text(
        0.075,
        0.015,
        "524,288 tokens | BF16 MHA 56x128 | K/V chunk 4,096 | "
        "filled markers are matched balanced reruns | 2026-08-24",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.055, 0.99, 0.98))
    save_svg(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the latest README benchmark figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_a30(args.output_dir / "latest-a30-host-memory-roofline.svg")
    plot_rtx5090(args.output_dir / "latest-rtx5090-host-memory-roofline.svg")


if __name__ == "__main__":
    main()
