"""Attribute a single-process Nsight SQLite projection capture through NVTX/CUDA correlation.

Use profiled timestamps for overlap diagnostics, not primary latency claims.
The capture must enable the producer's NVTX ranges. The output contains actual
GPU activity intervals, CPU synchronization calls, and no inferred memory data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(left, right):
    left, right = merged(left), merged(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = {r["id"]: r["value"] for r in db.execute("SELECT * FROM StringIds")}
        ranges = []
        counters = {}
        for row in db.execute("SELECT * FROM NVTX_EVENTS ORDER BY start"):
            item = dict(row)
            label = item["text"] or strings.get(item["textId"], "")
            if label not in {
                "seqattn:projection_hidden_h2d",
                "seqattn:qkv_projection",
                "seqattn:projection_qkv_d2h",
            }:
                continue
            item["label"] = label
            item["tile"] = counters.get(label, 0)
            counters[label] = item["tile"] + 1
            ranges.append(item)
        runtime = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
        by_correlation = {row["correlationId"]: row for row in runtime}
        if len(by_correlation) != len(runtime):
            raise ValueError("capture must have unambiguous single-process correlation IDs")

        def owner(api):
            matches = [
                r
                for r in ranges
                if r["end"] is not None
                and r["globalTid"] == api["globalTid"]
                and r["start"] <= api["start"] <= api["end"] <= r["end"]
            ]
            return min(matches, key=lambda r: r["end"] - r["start"]) if matches else None

        activities = []
        for table, kind in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ):
            for row in db.execute("SELECT * FROM " + table):
                api = by_correlation.get(row["correlationId"])
                region = None if api is None else owner(api)
                if region is None:
                    continue
                phase = region["label"].removeprefix("seqattn:")
                activities.append(
                    {
                        "start_ns": row["start"],
                        "end_ns": row["end"],
                        "stream": row["streamId"],
                        "tile": region["tile"],
                        "phase": phase,
                        "kind": kind,
                        "name": strings[row["shortName"]] if kind == "kernel" else "memcpy",
                        "bytes": row["bytes"] if kind == "memcpy" else None,
                        "copy_kind": row["copyKind"] if kind == "memcpy" else None,
                        "correlation_id": row["correlationId"],
                        "api_start_ns": api["start"],
                        "api_end_ns": api["end"],
                    }
                )
        if not activities:
            raise ValueError("capture has no attributable projection GPU activities")
        base = min(a["start_ns"] for a in activities)
        for item in activities:
            for key in ("start_ns", "end_ns", "api_start_ns", "api_end_ns"):
                item[key] -= base
        synchronization = []
        for row in runtime:
            name = strings[row["nameId"]]
            region = owner(row)
            if region and "Synchronize" in name:
                synchronization.append(
                    {
                        "name": name,
                        "phase": region["label"],
                        "tile": region["tile"],
                        "start_ns": row["start"] - base,
                        "end_ns": row["end"] - base,
                    }
                )
    compute = [
        (a["start_ns"], a["end_ns"])
        for a in activities
        if a["phase"] == "qkv_projection" and a["kind"] == "kernel"
    ]
    summary = {}
    for name, kind in (("pack", "kernel"), ("d2h", "memcpy")):
        intervals = [
            (a["start_ns"], a["end_ns"])
            for a in activities
            if a["phase"] == "projection_qkv_d2h" and a["kind"] == kind
        ]
        total = duration(intervals)
        overlap = intersection(intervals, compute)
        summary[name] = {
            "active_ms": total / 1e6,
            "overlap_compute_ms": overlap / 1e6,
            "overlap_fraction": overlap / total if total else None,
        }
    summary["gpu_span_ms"] = max(a["end_ns"] for a in activities) / 1e6
    summary["tile_counts"] = counters
    summary["projection_cpu_sync_count"] = len(synchronization)
    return {
        "source": "measured Nsight GPU activities; not unprofiled latency",
        "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "absolute_origin_ns": base,
        "summary": summary,
        "activities": sorted(activities, key=lambda a: a["start_ns"]),
        "cpu_synchronization": synchronization,
    }


def svg(data):
    width, left, scale = 1200, 150, 1000 / data["summary"]["gpu_span_ms"]
    rows = {"qkv_projection": 70, "pack": 125, "d2h": 180, "projection_hidden_h2d": 235}
    colors = ("#087f8c", "#bb5a3c", "#5968ad", "#64862f")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="345" viewBox="0 0 {width} 345">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#243748">',
        '<text x="20" y="25">RTX 5090: measured projection GPU timeline (8 x 4096 tokens)</text>',
    ]
    for label, y in rows.items():
        parts.append(f'<text x="10" y="{y + 18}">{label}</text>')
    for item in data["activities"]:
        phase = item["phase"]
        if phase == "projection_qkv_d2h":
            phase = "pack" if item["kind"] == "kernel" else "d2h"
        if phase == "qkv_projection" and item["kind"] != "kernel":
            continue
        x = left + item["start_ns"] / 1e6 * scale
        w = max(0.3, (item["end_ns"] - item["start_ns"]) / 1e6 * scale)
        color = colors[item["tile"] % len(colors)]
        parts.append(
            f'<rect x="{x:.3f}" y="{rows[phase]}" width="{w:.3f}" height="26" fill="{color}">'
            f"<title>tile {item['tile']}: {item['start_ns'] / 1e6:.3f}–"
            f"{item['end_ns'] / 1e6:.3f} ms</title></rect>"
        )
    for tick in range(0, int(data["summary"]["gpu_span_ms"]) + 1, 5):
        x = left + tick * scale
        parts.append(f'<text x="{x:.2f}" y="290">{tick}</text>')
    parts += [
        '<text x="1090" y="315">time (ms)</text>',
        '<text x="20" y="335">Colors identify tiles; gaps are observed. No memory allocation data is inferred.</text>',
        "</g></svg>",
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    data = analyze(args.sqlite)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    if args.svg:
        args.svg.write_text(svg(data))
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
