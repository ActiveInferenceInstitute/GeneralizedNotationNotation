"""Pins for ``utils/performance_tracking`` (previously 40% coverage)."""

from __future__ import annotations

import math

from gnn.utils.performance_tracking import (
    PerformanceTracker,
    get_performance_tracker,
)


def test_record_timing_and_summary() -> None:
    tracker = PerformanceTracker()

    tracker.record_timing("render", 1.5)
    tracker.record_timing("render", 2.5)
    tracker.record_timing("execute", 3.0, metadata={"steps": 25})

    summary = tracker.get_summary()

    assert summary["render"]["count"] == 2
    assert math.isclose(summary["render"]["total_duration"], 4.0)
    assert math.isclose(summary["render"]["avg_duration"], 2.0)
    assert math.isclose(summary["render"]["min_duration"], 1.5)
    assert math.isclose(summary["render"]["max_duration"], 2.5)
    assert summary["execute"]["count"] == 1


def test_track_operation_context_manager_records_duration() -> None:
    tracker = PerformanceTracker()

    with tracker.track_operation("w2_op", metadata={"tag": "contract"}):
        pass

    summary = tracker.get_summary()
    assert summary["w2_op"]["count"] == 1
    assert summary["w2_op"]["total_duration"] >= 0.0


def test_track_operation_records_duration_on_exception() -> None:
    tracker = PerformanceTracker()

    try:
        with tracker.track_operation("w2_failing"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert tracker.get_summary()["w2_failing"]["count"] == 1


def test_memory_tracking_monotonic_max() -> None:
    tracker = PerformanceTracker()

    tracker.record_memory_usage(50.0)
    tracker.record_memory_usage(120.5)
    tracker.record_memory_usage(80.0)

    assert math.isclose(tracker.max_memory_mb, 120.5)
    assert isinstance(tracker.current_memory_mb, float)


def test_get_performance_tracker_is_singleton() -> None:
    assert get_performance_tracker() is get_performance_tracker()
    assert isinstance(get_performance_tracker(), PerformanceTracker)


def test_timestamp_is_iso_format() -> None:
    tracker = PerformanceTracker()

    assert "T" in tracker.get_timestamp()
