"""Real production whole-block recordings are an independent behavioral oracle."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(ROOT / "benchmarks/audit_h3_runtime_behavior.py"))


@pytest.mark.parametrize("mode", ["materialized", "recompute"])
def test_complete_graph_matches_recorded_production_h3(mode):
    record = json.loads(
        (ROOT / f"docs/experiments/h3_full_behavior_20260909/{mode}.json").read_text()
    )
    spec = HELPER["build_recorded_shape"](record)
    result = HELPER["audit"](spec, record)
    assert result["status"] == "pass"
    assert result["compared_operator_events"] >= 30
    # Ensure this oracle checks behavior, rather than only accepting call counts.
    changed = copy.deepcopy(record)
    attention = next(r for r in changed["records"] if r["name"] == "attention")
    attention["initialize"] = not attention["initialize"]
    with pytest.raises(AssertionError, match="attention_boundaries"):
        HELPER["audit"](spec, changed)
