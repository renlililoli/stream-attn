from __future__ import annotations

import argparse
import json
import statistics
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
MULTIGPU_OBSERVATIONS = ROOT / "docs/experiments/rtx5090_dynamic_multigpu_524k_20260826"
MULTIGPU_FILES = {
    "2-GPU static": MULTIGPU_OBSERVATIONS / "two_gpu_static_tuned_final.json",
    "2-GPU dynamic": MULTIGPU_OBSERVATIONS / "two_gpu_dynamic.json",
    "3-GPU static": MULTIGPU_OBSERVATIONS / "three_gpu_static_tuned.json",
    "3-GPU dynamic": MULTIGPU_OBSERVATIONS / "three_gpu_dynamic.json",
}
HISTORICAL_SINGLE_GPU_SECONDS = 36.010153

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


def plot_rtx5090_multigpu(output: Path) -> None:
    observations = {name: load_json(path) for name, path in MULTIGPU_FILES.items()}
    seconds = {
        "2-GPU static": observations["2-GPU static"]["median_seconds"],
        "2-GPU dynamic": statistics.median(
            call["seconds"] for call in observations["2-GPU dynamic"]["calls"][-3:]
        ),
        "3-GPU static": observations["3-GPU static"]["median_seconds"],
        "3-GPU dynamic": statistics.median(
            call["seconds"] for call in observations["3-GPU dynamic"]["calls"][-3:]
        ),
    }
    speedup = {name: HISTORICAL_SINGLE_GPU_SECONDS / value for name, value in seconds.items()}

    gpu_counts = [1, 2, 3]
    static_speedup = [1.0, speedup["2-GPU static"], speedup["3-GPU static"]]
    dynamic_speedup = [1.0, speedup["2-GPU dynamic"], speedup["3-GPU dynamic"]]
    static_efficiency = [value / gpu * 100.0 for gpu, value in zip(gpu_counts, static_speedup)]
    dynamic_efficiency = [value / gpu * 100.0 for gpu, value in zip(gpu_counts, dynamic_speedup)]

    fig, (speed_ax, efficiency_ax) = plt.subplots(1, 2, figsize=(12.2, 5.5))
    speed_ax.plot(
        gpu_counts,
        gpu_counts,
        color=MODEL,
        linewidth=1.7,
        linestyle=(0, (5, 4)),
        label="Ideal linear scaling",
    )
    for values, color, marker, label in [
        (static_speedup, TEAL, "o", "Tuned static"),
        (dynamic_speedup, ORANGE, "s", "Converged dynamic"),
    ]:
        speed_ax.plot(
            gpu_counts,
            values,
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            linewidth=2.2,
            label=label,
            zorder=3,
        )
        for gpu, value in zip(gpu_counts[1:], values[1:]):
            speed_ax.annotate(
                f"{value:.2f}x",
                xy=(gpu, value),
                xytext=(0, -17 if label == "Tuned static" else 10),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
    speed_ax.set_title("Measured speedup", loc="left")
    speed_ax.set_xlabel("RTX 5090 GPUs")
    speed_ax.set_ylabel("Speedup vs historical 1-GPU SeqAttn")
    speed_ax.set_xticks(gpu_counts)
    speed_ax.set_xlim(0.85, 3.15)
    speed_ax.set_ylim(0.75, 3.15)
    finish_axes(speed_ax)
    speed_ax.legend(loc="upper left", fontsize=9)

    x = [2, 3]
    width = 0.3
    static_bars = efficiency_ax.bar(
        [value - width / 2 for value in x],
        static_efficiency[1:],
        width,
        color=TEAL,
        label="Tuned static",
    )
    dynamic_bars = efficiency_ax.bar(
        [value + width / 2 for value in x],
        dynamic_efficiency[1:],
        width,
        color=ORANGE,
        label="Converged dynamic",
    )
    efficiency_ax.axhline(100.0, color=MODEL, linewidth=1.5, linestyle=(0, (5, 4)))
    efficiency_ax.bar_label(static_bars, fmt="%.1f%%", padding=4, color=TEAL, fontweight="bold")
    efficiency_ax.bar_label(dynamic_bars, fmt="%.1f%%", padding=4, color=ORANGE, fontweight="bold")
    efficiency_ax.set_title("Parallel efficiency", loc="left")
    efficiency_ax.set_xlabel("RTX 5090 GPUs")
    efficiency_ax.set_ylabel("Efficiency")
    efficiency_ax.set_xticks(x)
    efficiency_ax.set_xticklabels(["2 GPUs", "3 GPUs"])
    efficiency_ax.set_ylim(0, 106)
    efficiency_ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0f}%")
    finish_axes(efficiency_ax)
    efficiency_ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "RTX 5090 multi-GPU scaling stays close to linear at 524K tokens",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.02,
        "524,288 tokens | BF16 MHA 56x128 | host output | tuned static Q | "
        "dynamic median after convergence | 2026-08-26",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.99, 0.91), w_pad=3.0)
    save_svg(fig, output, date="2026-08-26")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the latest README benchmark figures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_rtx5090_multigpu(args.output_dir / "latest-rtx5090-multigpu-efficiency.svg")
    plot_a30(args.output_dir / "latest-a30-host-memory-roofline.svg")
    plot_rtx5090(args.output_dir / "latest-rtx5090-host-memory-roofline.svg")


if __name__ == "__main__":
    main()
