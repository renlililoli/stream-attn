"""Self-contained HTML and versioned JSON export; no network or plotting dependencies."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ._report_template import HTML_TEMPLATE
from .memory import memory_statistics
from .search import ActivationMemoryEstimate
from .specs import ExecutionTrace


def timeline_report_data(result: ExecutionTrace | ActivationMemoryEstimate) -> dict:
    if isinstance(result, ExecutionTrace):
        candidates = [{"trace": asdict(result), "stats": asdict(memory_statistics(result))}]
        selection = {"selected_index": 0}
    else:
        candidates = [
            {
                "trace": asdict(item.trace),
                "stats": asdict(item.stats),
                "objective_peak_bytes": item.peak_bytes,
            }
            for item in result.candidates
        ]
        index = {id(item): i for i, item in enumerate(result.candidates)}
        selection = {
            "selected_index": index[id(result.selected)] if result.selected else None,
            "minimum_capacity_index": index[id(result.minimum_capacity)]
            if result.minimum_capacity
            else None,
            "pareto_indices": [index[id(item)] for item in result.pareto],
            "objective_pool": result.candidates[0].objective_pool,
            "objective_owners": sorted(result.candidates[0].objective_owners)
            if result.candidates[0].objective_owners is not None
            else None,
            "latency_limit_seconds": result.latency_limit_seconds,
            "reference_latency_seconds": result.reference_latency_seconds,
            "target_throughput_fraction": result.target_throughput_fraction,
        }
    return {"schema_version": 1, "selection": selection, "candidates": candidates}


def write_timeline_report(
    result: ExecutionTrace | ActivationMemoryEstimate,
    html_path: str | Path,
    *,
    json_path: str | Path | None = None,
) -> None:
    """Write an offline interactive report, plus optional canonical JSON.

    Inputs, including untrusted profile labels, are data only. No telemetry,
    external scripts/fonts, or runtime accelerator instrumentation is used.
    """
    data = timeline_report_data(result)
    Path(html_path).write_text(render_timeline_html(data), encoding="utf-8")
    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )


def render_timeline_html(data: dict) -> str:
    """Render canonical report data as standalone HTML without filesystem I/O."""
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    safe_payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return HTML_TEMPLATE.replace("__REPORT_JSON__", safe_payload)
