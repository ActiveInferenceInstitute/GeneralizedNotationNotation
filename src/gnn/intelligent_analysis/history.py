#!/usr/bin/env python3
"""
Run-history analysis for the intelligent analysis module.

Provides pure functions that compare a current pipeline execution receipt
against one or more previous receipts, producing run-over-run deltas and a
trend classification. This implements the module roadmap item "Historical
trend analysis across multiple pipeline runs" without introducing any I/O:
callers pass already-loaded summary dictionaries, keeping the functions
deterministic and trivially testable.

Typical use::

    from gnn.intelligent_analysis.history import analyze_run_history

    history = analyze_run_history(current_summary, [prev_run_1, prev_run_2])
    print(history["trend"]["direction"])          # "improving" | ...
    print(history["current"]["health_score"])     # float 0-100
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .analyzer import calculate_pipeline_health_score, pick_evidence_timestamp

__all__ = [
    "RunSnapshot",
    "StepDelta",
    "build_run_snapshot",
    "compute_step_deltas",
    "classify_trend",
    "analyze_run_history",
]


@dataclass
class RunSnapshot:
    """Normalized metrics extracted from one pipeline execution summary."""

    overall_status: str
    health_score: float
    total_duration_seconds: float
    peak_memory_mb: float
    step_count: int
    failure_count: int
    successful_count: int
    warning_count: int
    timestamp: str = "unavailable"
    script_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "overall_status": self.overall_status,
            "health_score": self.health_score,
            "total_duration_seconds": self.total_duration_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "step_count": self.step_count,
            "failure_count": self.failure_count,
            "successful_count": self.successful_count,
            "warning_count": self.warning_count,
            "timestamp": self.timestamp,
        }
        if self.script_name is not None:
            data["script_name"] = self.script_name
        return data


@dataclass
class StepDelta:
    """Duration/memory change for one step name between two runs."""

    script_name: str
    duration_delta_seconds: float
    duration_ratio: Optional[float]
    memory_delta_mb: float
    memory_ratio: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "script_name": self.script_name,
            "duration_delta_seconds": self.duration_delta_seconds,
            "duration_ratio": self.duration_ratio,
            "memory_delta_mb": self.memory_delta_mb,
            "memory_ratio": self.memory_ratio,
        }


def build_run_snapshot(
    summary_data: Dict[str, Any], script_name: Optional[str] = None
) -> RunSnapshot:
    """Extract normalized metrics from one pipeline execution summary.

    Args:
        summary_data: Pipeline execution summary dictionary
        script_name: Optional label identifying the run (e.g. a receipt path)

    Returns:
        A :class:`RunSnapshot` of the run's headline metrics
    """
    steps: List[Dict[str, Any]] = summary_data.get("steps", [])
    perf: Dict[str, Any] = summary_data.get("performance_summary", {})
    statuses = [str(step.get("status", "")).upper() for step in steps]
    return RunSnapshot(
        overall_status=str(summary_data.get("overall_status", "UNKNOWN")),
        health_score=calculate_pipeline_health_score(summary_data),
        total_duration_seconds=float(summary_data.get("total_duration_seconds", 0)),
        peak_memory_mb=float(perf.get("peak_memory_mb", 0)),
        step_count=len(steps),
        failure_count=sum(status == "FAILED" for status in statuses),
        successful_count=sum(status.startswith("SUCCESS") for status in statuses),
        warning_count=sum("WARNING" in status for status in statuses),
        timestamp=pick_evidence_timestamp(summary_data),
        script_name=script_name,
    )


def compute_step_deltas(
    current: Dict[str, Any], previous: Dict[str, Any]
) -> List[StepDelta]:
    """Compare per-step duration and memory between two runs.

    Steps are matched by ``script_name``; steps present in only one run are
    omitted rather than inventing baselines.

    Args:
        current: Pipeline execution summary for the current run
        previous: Pipeline execution summary for the previous run

    Returns:
        List of :class:`StepDelta` sorted by absolute duration delta,
        descending
    """
    prev_by_name = {
        step.get("script_name"): step
        for step in previous.get("steps", [])
        if step.get("script_name")
    }
    deltas: list[StepDelta] = []
    for step in current.get("steps", []):
        name = step.get("script_name")
        if not name or name not in prev_by_name:
            continue
        prev_step = prev_by_name[name]
        cur_duration = step.get("duration_seconds", 0)
        prev_duration = prev_step.get("duration_seconds", 0)
        cur_memory = step.get("peak_memory_mb", 0)
        prev_memory = prev_step.get("peak_memory_mb", 0)
        deltas.append(
            StepDelta(
                script_name=str(name),
                duration_delta_seconds=cur_duration - prev_duration,
                duration_ratio=(cur_duration / prev_duration)
                if prev_duration > 0
                else None,
                memory_delta_mb=cur_memory - prev_memory,
                memory_ratio=(cur_memory / prev_memory) if prev_memory > 0 else None,
            )
        )
    deltas.sort(key=lambda d: abs(d.duration_delta_seconds), reverse=True)
    return deltas


def classify_trend(
    current_score: float, previous_scores: List[float]
) -> Dict[str, Any]:
    """Classify the health-score trend across runs.

    Args:
        current_score: Health score (0-100) of the current run
        previous_scores: Health scores of previous runs, oldest first

    Returns:
        Dictionary with ``direction`` ("improving" | "degrading" |
        "stable" | "insufficient_history"), ``mean_previous`` (or None),
        and ``delta`` (current minus mean previous, or None)
    """
    if not previous_scores:
        return {
            "direction": "insufficient_history",
            "mean_previous": None,
            "delta": None,
        }
    mean_previous = sum(previous_scores) / len(previous_scores)
    delta = current_score - mean_previous
    if delta > 5.0:
        direction = "improving"
    elif delta < -5.0:
        direction = "degrading"
    else:
        direction = "stable"
    return {
        "direction": direction,
        "mean_previous": mean_previous,
        "delta": delta,
    }


def analyze_run_history(
    current_summary: Dict[str, Any],
    previous_summaries: List[Dict[str, Any]],
    current_label: Optional[str] = None,
    max_deltas: int = 5,
) -> Dict[str, Any]:
    """Analyze a pipeline run against its run-over-run history.

    Args:
        current_summary: Pipeline execution summary for the current run
        previous_summaries: Summaries of previous runs, oldest first
        current_label: Optional label for the current run
        max_deltas: Maximum number of per-step deltas to include (by
            absolute duration change)

    Returns:
        Dictionary with ``current`` snapshot, ``previous`` snapshots,
        ``trend`` classification, and ``step_deltas``
    """
    previous_snapshots = [build_run_snapshot(prev) for prev in previous_summaries]
    current_snapshot = build_run_snapshot(current_summary, current_label)
    deltas = compute_step_deltas(current_summary, previous_summaries[-1])
    return {
        "current": current_snapshot.to_dict(),
        "previous": [snap.to_dict() for snap in previous_snapshots],
        "trend": classify_trend(
            current_snapshot.health_score,
            [snap.health_score for snap in previous_snapshots],
        ),
        "step_deltas": [d.to_dict() for d in deltas[:max_deltas]],
    }
