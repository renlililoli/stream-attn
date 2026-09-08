import json
from dataclasses import replace

import pytest

from seqattn_core.estimation import (
    BufferSpec,
    ExecutionSpec,
    MemoryPool,
    OperationSpec,
    RateProfile,
    estimate_activation_memory,
    memory_statistics,
    schedule_execution,
    trace_from_measurements,
    write_timeline_report,
)


def operation(name, seconds, *, resources=("matrix",), dependencies=(), kind="compute"):
    return OperationSpec(name, seconds, resources, dependencies, kind)


def execution(operations, buffers=(), pools=None, name="example"):
    pools = (MemoryPool("device"),) if pools is None else pools
    return ExecutionSpec(name, pools, tuple(operations), tuple(buffers))


def test_scheduler_fills_resource_gaps_without_violating_dependencies():
    spec = execution(
        [
            operation("copy", 10, resources=("dma",), kind="io"),
            operation("after-copy", 2, dependencies=("copy",)),
            operation("independent", 3),
            operation(
                "all-resources", 1, resources=("dma", "matrix"), dependencies=("independent",)
            ),
        ]
    )
    trace = schedule_execution(spec)
    times = {op.name: (op.start_seconds, op.end_seconds) for op in trace.operations}
    assert times == {
        "copy": (0, 10),
        "after-copy": (10, 12),
        "independent": (0, 3),
        "all-resources": (12, 13),
    }
    stats = memory_statistics(trace)
    assert stats.compute_io_overlap_seconds == 3
    assert stats.resource_busy_seconds == {"dma": 11, "matrix": 6}
    assert trace.duration_seconds == 13


def test_single_resource_serializes_compute_and_io():
    spec = execution(
        [
            operation("copy", 2, resources=("shared",), kind="io"),
            operation("compute", 3, resources=("shared",)),
        ]
    )
    trace = schedule_execution(spec)
    assert trace.duration_seconds == 5
    assert memory_statistics(trace).compute_io_overlap_seconds == 0


def test_dependency_cycle_and_unknown_names_fail():
    with pytest.raises(ValueError, match="cycle"):
        schedule_execution(
            execution(
                [operation("a", 1, dependencies=("b",)), operation("b", 1, dependencies=("a",))]
            )
        )
    with pytest.raises(ValueError, match="unknown dependency"):
        execution([operation("a", 1, dependencies=("missing",))])
    with pytest.raises(ValueError, match="unknown pool or operation"):
        execution([operation("a", 1)], [BufferSpec("x", "device", 1, "Q", ("missing",))])


def test_peak_is_simultaneous_occupancy_and_persistent_buffers_stay_allocated():
    spec = execution(
        [operation("a", 2), operation("b", 2, dependencies=("a",))],
        [
            BufferSpec("resident", "device", 100, "state", ("a",), persistent=True),
            BufferSpec("first", "device", 60, "Q", ("a",)),
            BufferSpec("second", "device", 90, "KV", ("b",)),
        ],
    )
    stats = memory_statistics(schedule_execution(spec)).pools[0]
    assert stats.peak_bytes == 190
    assert stats.peak_seconds == 2
    assert stats.components_at_peak == {"KV": 90, "Q": 0, "state": 100}
    assert sum(stats.component_peak_bytes.values()) == 250
    assert stats.average_bytes == 175
    assert stats.byte_seconds == 700
    assert [(p.seconds, p.total_bytes, p.active_bytes) for p in stats.points] == [
        (0, 160, 160),
        (2, 190, 90),
        (4, 100, 0),
    ]


def test_alignment_distinct_pools_and_async_copy_lifetime():
    spec = execution(
        [
            operation("gemm", 1),
            operation("copy", 2, resources=("dma",), dependencies=("gemm",), kind="io"),
            operation("later", 4, dependencies=("gemm",)),
        ],
        [
            BufferSpec("projected", "device", 257, "QKV", ("gemm", "copy")),
            BufferSpec("host", "host", 100, "QKV", ("copy",), persistent=True, owner="caller"),
        ],
        pools=(MemoryPool("device", 511, 256), MemoryPool("host")),
    )
    trace = schedule_execution(spec)
    buffer = trace.buffers[0]
    assert (buffer.start_seconds, buffer.end_seconds, buffer.allocated_bytes) == (0, 3, 512)
    stats = memory_statistics(trace)
    assert [p.peak_bytes for p in stats.pools] == [512, 100]
    assert stats.pools[0].exceeds_capacity
    assert not stats.fits_capacity


def test_alias_uses_count_one_physical_allocation_and_active_intervals_are_unioned():
    spec = execution(
        [
            operation("read-a", 2, resources=("a",)),
            operation("read-b", 3, resources=("b",)),
            operation("later", 2, dependencies=("read-a", "read-b")),
        ],
        [BufferSpec("shared", "device", 128, "activation", ("read-a", "read-b", "later"))],
    )
    trace = schedule_execution(spec)
    assert trace.buffers[0].active_intervals == ((0, 5),)
    stats = memory_statistics(trace).pools[0]
    assert stats.peak_bytes == stats.active_peak_bytes == 128


def test_memory_sweep_matches_independent_midpoint_sum():
    operations = [operation(f"op{i}", i + 1, resources=(f"r{i % 3}",)) for i in range(12)]
    buffers = [
        BufferSpec(
            f"b{i}",
            "device",
            13 * (i + 1),
            f"c{i % 3}",
            (f"op{i}", f"op{11 - i}") if i != 11 - i else (f"op{i}",),
            persistent=i % 4 == 0,
        )
        for i in range(12)
    ]
    trace = schedule_execution(
        execution(operations, buffers, pools=(MemoryPool("device", None, 16),))
    )
    stats = memory_statistics(trace).pools[0]
    area = 0
    for left, right in zip(stats.points, stats.points[1:], strict=False):
        middle = (left.seconds + right.seconds) / 2
        live = [b for b in trace.buffers if b.start_seconds <= middle < b.end_seconds]
        assert left.total_bytes == sum(b.allocated_bytes for b in live)
        active = [b for b in trace.buffers if any(a <= middle < z for a, z in b.active_intervals)]
        assert left.active_bytes == sum(b.allocated_bytes for b in active)
        area += left.total_bytes * (right.seconds - left.seconds)
    assert area == stats.byte_seconds


def test_candidate_selection_uses_whole_trace_and_retains_infeasible_fast_reference():
    candidates = [
        execution(
            [operation("work", duration)],
            [
                BufferSpec("activation", "device", size, "activation", ("work",)),
            ],
            pools=(MemoryPool("device", 150),),
            name=name,
        )
        for name, duration, size in (("small", 10, 100), ("balanced", 8.2, 150), ("fast", 8, 200))
    ]
    result = estimate_activation_memory(candidates, objective_pool="device")
    assert result.minimum_capacity.trace.name == "small"
    assert result.selected.trace.name == "balanced"
    assert result.reference_latency_seconds == 8
    assert [p.trace.name for p in result.pareto] == ["small", "balanced"]
    assert not result.candidates[2].stats.fits_capacity
    impossible = estimate_activation_memory(
        candidates, objective_pool="device", max_latency_seconds=7
    )
    assert impossible.selected is None
    assert len(impossible.candidates) == 3


def test_measured_timestamps_rebuild_lifetimes_and_reject_inconsistent_clocks():
    spec = execution(
        [operation("a", 1), operation("b", 1, dependencies=("a",))],
        [BufferSpec("temp", "device", 64, "Q", ("a", "b"))],
    )
    trace = trace_from_measurements(
        spec, {"a": (0.5, 2), "b": (3, 4)}, provenance="device event export"
    )
    assert trace.source == "measured"
    assert (trace.buffers[0].start_seconds, trace.buffers[0].end_seconds) == (0.5, 4)
    assert trace.buffers[0].active_intervals == ((0.5, 2), (3, 4))
    with pytest.raises(ValueError, match="dependency"):
        trace_from_measurements(spec, {"a": (0, 2), "b": (1, 3)}, provenance="bad clock")
    with pytest.raises(ValueError, match="exactly"):
        trace_from_measurements(spec, {"a": (0, 2)}, provenance="incomplete")


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True])
def test_invalid_hardware_rates_fail(rate):
    with pytest.raises((TypeError, ValueError)):
        RateProfile("GEMM", rate, ("matrix",))


def test_report_json_is_complete_and_html_labels_cannot_break_out_of_data(tmp_path):
    attack = "</script><script>globalThis.injected=true</script> & <unsafe>"
    spec = replace(execution([operation("a", 1)]), name=attack)
    trace = schedule_execution(spec)
    report, payload = tmp_path / "report.html", tmp_path / "report.json"
    write_timeline_report(trace, report, json_path=payload)
    data = json.loads(payload.read_text())
    assert data["schema_version"] == 1
    assert data["candidates"][0]["trace"]["name"] == attack
    text = report.read_text()
    assert attack not in text
    embedded = text.split('<script id="report-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == data
    assert 'src="http' not in text
    assert "Export SVG" in text


def test_resource_gap_lookup_matches_exhaustive_search():
    from seqattn_core.estimation.schedule import _first_gap

    calendars = {"compute": [(0, 2), (6, 9), (15, 22)], "dma": [(1, 4), (11, 14), (17, 20)]}
    for start in range(25):
        for duration in range(1, 8):
            expected = next(
                candidate
                for candidate in range(start, 40)
                if all(
                    candidate + duration <= left or candidate >= right
                    for intervals in calendars.values()
                    for left, right in intervals
                )
            )
            assert _first_gap(calendars, ("compute", "dma"), start, duration) == expected


def test_lifetime_anchors_keep_idle_callback_results_until_release_milestone():
    spec = execution(
        [
            operation("produce", 1),
            operation("other", 2, dependencies=("produce",)),
            OperationSpec("release", 0, (), ("other",), kind="control"),
        ],
        [BufferSpec("kept", "device", 128, "QKV", ("produce",), release_after="release")],
    )
    trace = schedule_execution(spec)
    assert trace.buffers[0].end_seconds == 3
    assert trace.buffers[0].active_intervals == ((0, 1),)
    assert memory_statistics(trace).pools[0].average_bytes == 128


def test_activation_objective_excludes_weights_but_capacity_still_counts_them():
    specs = [
        execution(
            [operation("work", 1)],
            [
                BufferSpec(
                    "activation", "device", activation, "activation", ("work",), owner="callback"
                ),
                BufferSpec("weight", "device", weight, "weights", ("work",), owner="weights"),
            ],
            pools=(MemoryPool("device", 180),),
            name=name,
        )
        for name, activation, weight in (("small-activation", 50, 150), ("fits", 100, 50))
    ]
    result = estimate_activation_memory(
        specs, objective_pool="device", objective_owners=frozenset({"callback"})
    )
    assert result.candidates[0].peak_bytes == 50
    assert not result.candidates[0].stats.fits_capacity
    assert result.selected.trace.name == "fits"
