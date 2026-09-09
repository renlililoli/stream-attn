"""Compare the entire H3 graph with independently recorded real runner decisions.

This checks behavioral signatures, not latency. The recording is made by wrapping
real production callbacks/kernels; no expected call list is supplied to the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    MemoryPool,
    RateProfile,
    build_h3_block_execution,
    schedule_execution,
)


def build_recorded_shape(record):
    shape = H3BlockShape(**{**record["shape"], "segments": tuple(record["shape"]["segments"])})
    config = H3ExecutionConfig(**record["config"])
    compute = RateProfile("behavior test only", 1e12, ("compute",))
    copy = RateProfile("behavior test only", 1e10, ("copy",))
    profile = H3DeviceProfile.from_rates(
        "behavior-only; not a latency calibration",
        device_pool=MemoryPool("device"),
        host_pool=MemoryPool("host"),
        attention=compute,
        gemm=compute,
        vector=compute,
        h2d=copy,
        d2h=copy,
        d2d=copy,
    )
    return build_h3_block_execution(
        shape,
        config,
        profile,
        callbacks=H3CallbackConfig(variant="modulated", linear_memory="profile"),
    )


def audit(spec, record):
    checks = []

    def equal(name, predicted, measured):
        if predicted != measured:
            raise AssertionError(f"{name}: predicted={predicted!r}, actual={measured!r}")
        checks.append(name)

    def selected(name):
        return [r for r in record["records"] if r["name"] == name]

    actual = []
    for event in record["records"]:
        key = event["name"]
        if key in {"adaln", "norm1", "qkv", "out", "norm2", "fc1", "swiglu_fc2"}:
            actual.append((key, event["shape"][0]))
        elif key == "project_rows":
            actual.append(("q" if event["row_start"] == 0 else "kv", event["shape"][0]))
        elif key in {"attention", "finalize"}:
            actual.append((key, event.get("q_tokens", event.get("tokens"))))
    predicted = []
    keys = {
        "adaln",
        "norm1",
        "qkv",
        "q",
        "kv",
        "out",
        "norm2",
        "fc1",
        "swiglu_fc2",
        "attention",
        "finalize",
    }
    for op in spec.operations:
        if op.component in keys:
            tokens = int(re.search(r"tokens=(\d+);", op.details)[1])
            predicted.append((op.component, tokens))
    equal("complete_operator_order_and_token_shapes", predicted, actual)
    meta = spec.metadata
    q_ranges = meta["query_ranges"]
    equal(
        "epilogue_ranges",
        [(a, b) for a, b, _ in q_ranges],
        [(r["start"], r["stop"]) for r in selected("epilogue")],
    )
    equal(
        "ffn_ranges_including_final_flush",
        meta["ffn_ranges"],
        [(r["start"], r["stop"]) for r in selected("ffn")],
    )
    equal(
        "ffn_carry_vs_direct_and_output_slot",
        [
            ("carry" if r["source"] == "ffn.carry" else "post", r["slot"])
            for r in meta["ffn_sources"]
        ],
        [(r["source"], r["slot"]) for r in selected("emit")],
    )
    equal(
        "ffn_returns_input_alias",
        [True] * len(meta["ffn_ranges"]),
        [r["aliases_input"] for r in selected("ffn_result")],
    )
    equal(
        "finalize_aliases_resident_q",
        [True] * len(q_ranges),
        [r["aliases_q"] for r in selected("finalize")],
    )
    equal(
        "epilogue_residual_is_new_device_storage",
        [False] * len(q_ranges),
        [r["aliases_host"] for r in selected("epilogue_result")],
    )
    expected_updates = []
    segments = record["shape"]["segments"]
    for query, start, stop, segment in meta["kv_ranges"]:
        q_start, q_stop, _ = q_ranges[query]
        offset = sum(segments[:segment])
        expected_updates.append(
            (
                q_stop - q_start,
                stop - start,
                q_start - offset,
                start - offset,
                start == offset,
                False,
            )
        )
    equal(
        "attention_boundaries_offsets_and_initialize",
        expected_updates,
        [
            (
                r["q_tokens"],
                r["kv_tokens"],
                r["q_local_offset"],
                r["kv_local_offset"],
                r["initialize"],
                r["causal"],
            )
            for r in selected("attention")
        ],
    )
    mode = record["mode"]
    equal("hidden_output_alias_policy", mode == "materialized", record["output_aliases_host"])
    equal("output_finite", True, record["finite"])
    stats = record["stats"]
    equal(
        "core_workspace_budget",
        meta["core_workspace_budget_bytes"],
        stats["estimated_workspace_bytes"],
    )
    equal("ffn_cross_q_boundaries", meta["ffn_cross_q_boundaries"], stats["ffn_cross_q_boundaries"])
    h, a, size = (
        record["shape"]["hidden_features"],
        record["shape"]["heads"] * record["shape"]["head_dim"],
        record["shape"]["element_bytes"],
    )
    equal(
        "final_hidden_d2h_bytes",
        sum(b - a for a, b in meta["ffn_ranges"]) * h * size,
        stats["final_hidden_d2h_bytes"],
    )
    if mode == "materialized":
        equal(
            "projection_ranges",
            meta["projection_ranges"],
            [(r["start"], r["stop"]) for r in selected("project_qkv")],
        )
        for index, result in enumerate(selected("qkv_result")):
            equal(f"qkv{index}_shares_one_backing_allocation", True, result["one_storage"])
            for tensor, contiguous in zip("QKV", result["contiguous"]):
                exists = any(o.name == f"projection{index}.{tensor}.pack" for o in spec.operations)
                equal(f"qkv{index}_{tensor}_packing_matches_actual_strides", not contiguous, exists)
        transferred = sum(b - a for a, b, _ in q_ranges) + 2 * sum(
            b - a for _, a, b, _ in meta["kv_ranges"]
        )
        equal(
            "attention_h2d_bytes",
            transferred * a * size,
            stats["projection"]["attention"]["h2d_bytes"],
        )
        equal(
            "projection_qkv_d2h_bytes",
            3 * sum(segments) * a * size,
            stats["projection"]["projection_qkv_d2h_bytes"],
        )
    else:
        equal(
            "recompute_q_ranges",
            [(a, b) for a, b, _ in q_ranges],
            [(r["start"], r["stop"]) for r in selected("project_q")],
        )
        equal(
            "recompute_kv_ranges",
            [(a, b) for _, a, b, _ in meta["kv_ranges"]],
            [(r["start"], r["stop"]) for r in selected("project_kv")],
        )
        rows = sum(b - a for a, b, _ in q_ranges) + sum(b - a for _, a, b, _ in meta["kv_ranges"])
        equal("recompute_hidden_h2d_bytes", rows * h * size, stats["recompute"]["hidden_h2d_bytes"])
        equal("recompute_no_host_qkv", 0, stats["recompute"]["qkv_host_bytes"])
    schedule_execution(spec)  # Also checks declared lifetimes cover every use.
    return {
        "mode": mode,
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "compared_operator_events": len(actual),
        "scope": "recorded production callback/operator order, ranges, aliases, transfer sizes and core workspace; not an exhaustive kernel/allocator trace",
    }


def audit_host_sync(spec, path):
    """Use NVTX attribution to verify the real blocking callback copies."""
    expected = sum(spec.metadata["operator_counts"].get(key, 0) for key in ("qkv", "q", "kv"))
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        names = {
            "seqattn:qkv_projection",
            "seqattn:recompute_q_projection",
            "seqattn:recompute_kv_projection",
        }
        regions = [
            r
            for r in db.execute("SELECT * FROM NVTX_EVENTS")
            if (r["text"] or strings.get(r["textId"], "")) in names
        ]
        count = 0
        for api in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if "StreamSynchronize" not in strings[api["nameId"]]:
                continue
            if any(
                r["end"]
                and r["globalTid"] == api["globalTid"]
                and r["start"] <= api["start"] <= api["end"] <= r["end"]
                for r in regions
            ):
                count += 1
    if count != expected or len(regions) != expected:
        raise AssertionError(
            f"blocking callback copies: expected {expected}, observed {count}, regions {len(regions)}"
        )
    return {"expected": expected, "observed": count, "status": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sqlite", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    spec = build_recorded_shape(record)
    result = audit(spec, record)
    if args.sqlite:
        result["host_sync"] = audit_host_sync(spec, args.sqlite)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
