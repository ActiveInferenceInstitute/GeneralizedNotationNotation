#!/usr/bin/env python3
"""
Functional tests for the Report module's diff-aware pipeline reporting and
semantic-fidelity / cross-framework-reliability ledger renderers.

Covers:
  - StepDiff / DiffReport diffing between two pipeline_run summaries
  - Regression, fixed-step, and timing-regression classification
  - archive_run() lifecycle (archive, prune, get_previous_run)
  - semantic-fidelity and cross-framework-reliability Markdown rendering
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from report.cross_framework_reliability import (  # noqa: E402
    render_cross_framework_reliability_markdown,
)
from report.diff_report import (  # noqa: E402
    DiffReport,
    StepDiff,
    archive_run,
    compare_runs,
    get_previous_run,
)
from report.semantic_fidelity import render_semantic_fidelity_markdown  # noqa: E402


def _write_summary(path: Path, timestamp: str, steps: list[dict[str, Any]]) -> None:
    """Write a pipeline_execution_summary.json fixture."""
    data = {"timestamp": timestamp, "overall_status": "SUCCESS", "steps": steps}
    path.write_text(json.dumps(data))


def _step(name: str, status: str, duration: float) -> dict[str, Any]:
    return {"name": name, "status": status, "duration_seconds": duration}


# ---------------------------------------------------------------------------
# compare_runs / DiffReport
# ---------------------------------------------------------------------------


class TestCompareRuns:
    """Test diff-aware run comparison."""

    @pytest.mark.unit
    def test_identical_runs_green_badge(self, tmp_path: Path) -> None:
        """Two identical successful runs should produce no regressions."""
        cur = tmp_path / "current.json"
        prev = tmp_path / "previous.json"
        _write_summary(cur, "2026-01-02", [_step("Step 3", "SUCCESS", 1.5)])
        _write_summary(prev, "2026-01-01", [_step("Step 3", "SUCCESS", 1.5)])
        report = compare_runs(cur, prev)
        assert report.overall_badge == "🟢"
        assert report.new_failures == []
        assert report.fixed_steps == []
        assert report.timing_regressions == []

    @pytest.mark.unit
    def test_new_failure_sets_red_badge(self, tmp_path: Path) -> None:
        """A step that newly fails should set the red badge."""
        cur = tmp_path / "current.json"
        prev = tmp_path / "previous.json"
        _write_summary(cur, "2026-01-02", [_step("Step 10", "FAILED", 2.0)])
        _write_summary(prev, "2026-01-01", [_step("Step 10", "SUCCESS", 1.0)])
        report = compare_runs(cur, prev)
        assert report.overall_badge == "🔴"
        assert report.new_failures == ["Step 10"]

    @pytest.mark.unit
    def test_fixed_step_detected(self, tmp_path: Path) -> None:
        """A step fixed between runs should appear in fixed_steps."""
        cur = tmp_path / "current.json"
        prev = tmp_path / "previous.json"
        _write_summary(cur, "2026-01-02", [_step("Step 5", "SUCCESS", 1.0)])
        _write_summary(prev, "2026-01-01", [_step("Step 5", "FAILED", 1.0)])
        report = compare_runs(cur, prev)
        assert "Step 5" in report.fixed_steps

    @pytest.mark.unit
    def test_timing_regression_sets_yellow_badge(self, tmp_path: Path) -> None:
        """A >20% slowdown without failure should set the yellow badge."""
        cur = tmp_path / "current.json"
        prev = tmp_path / "previous.json"
        _write_summary(cur, "2026-01-02", [_step("Step 7", "SUCCESS", 30.0)])
        _write_summary(prev, "2026-01-01", [_step("Step 7", "SUCCESS", 10.0)])
        report = compare_runs(cur, prev)
        assert report.overall_badge == "🟡"
        assert len(report.timing_regressions) == 1

    @pytest.mark.unit
    def test_steps_added_or_removed_handled(self, tmp_path: Path) -> None:
        """Steps present in only one run should be handled gracefully."""
        cur = tmp_path / "current.json"
        prev = tmp_path / "previous.json"
        _write_summary(
            cur, "2026-01-02", [_step("Step 3", "SUCCESS", 1.0), _step("Step 4", "SUCCESS", 2.0)]
        )
        _write_summary(prev, "2026-01-01", [_step("Step 3", "SUCCESS", 1.0)])
        report = compare_runs(cur, prev)
        names = {sd.step_name for sd in report.step_diffs}
        assert "Step 3" in names
        assert "Step 4" in names

    @pytest.mark.unit
    def test_missing_files_return_unknown_report(self, tmp_path: Path) -> None:
        """Missing summary files should yield an 'unknown' report, not raise."""
        report = compare_runs(tmp_path / "nope.json", tmp_path / "missing.json")
        assert report.current_timestamp == "unknown"
        assert report.step_diffs == []


class TestDiffReportMarkdown:
    """Render DiffReport to Markdown."""

    @pytest.mark.unit
    def test_to_markdown_contains_sections(self) -> None:
        """to_markdown should include the badge and comparison blurb."""
        report = DiffReport(
            current_timestamp="now",
            previous_timestamp="past",
            step_diffs=[
                StepDiff(
                    step_name="Step 9",
                    prev_status="SUCCESS",
                    curr_status="FAILED",
                    prev_duration=1.0,
                    curr_duration=2.0,
                    duration_delta_pct=100.0,
                    is_regression=True,
                )
            ],
            new_failures=["Step 9"],
        )
        md = report.to_markdown()
        assert "Run Comparison" in md
        assert "now" in md and "past" in md
        assert "Step 9" in md
        assert "New Failures" in md


class TestArchiveRun:
    """Archive lifecycle helpers."""

    @pytest.mark.unit
    def test_archive_run_copies_summary(self, tmp_path: Path) -> None:
        """archive_run should copy the summary into a .history dir."""
        summary = tmp_path / "pipeline_execution_summary.json"
        _write_summary(summary, "2026-01-01", [])
        archived = archive_run(summary)
        assert archived is not None
        assert archived.exists()
        assert archived.parent.name == ".history"

    @pytest.mark.unit
    def test_archive_run_missing_source_returns_none(self, tmp_path: Path) -> None:
        """archive_run with a missing source should return None."""
        assert archive_run(tmp_path / "missing.json") is None

    @pytest.mark.unit
    def test_archive_run_prunes_old_archives(self, tmp_path: Path) -> None:
        """archive_run should prune archives above max_archives."""
        summary = tmp_path / "summary.json"
        _write_summary(summary, "2026-01-01", [])
        for _ in range(3):
            archive_run(summary, max_archives=2)
        history_dir = tmp_path / ".history"
        archives = list(history_dir.glob("*.json"))
        assert len(archives) <= 2

    @pytest.mark.unit
    def test_get_previous_run_returns_newest(self, tmp_path: Path) -> None:
        """get_previous_run should return the lexicographically newest archive."""
        history = tmp_path / ".history"
        history.mkdir()
        (history / "20260101_000000.json").write_text("{}")
        (history / "20260102_000000.json").write_text("{}")
        prev = get_previous_run(history)
        assert prev is not None
        assert "20260102" in prev.name

    @pytest.mark.unit
    def test_get_previous_run_empty_history(self, tmp_path: Path) -> None:
        """get_previous_run with no history should return None."""
        assert get_previous_run(tmp_path / ".history") is None


# ---------------------------------------------------------------------------
# Markdown ledger renderers
# ---------------------------------------------------------------------------


class TestSemanticFidelityRenderer:
    """Render the semantic-fidelity ledger to Markdown."""

    @pytest.mark.unit
    def test_renders_ledger_rows(self) -> None:
        """A populated ledger should produce family and round-trip rows."""
        ledger: dict[str, Any] = {
            "schema": "gnn-semantic-fidelity-v1",
            "family_count": 1,
            "strict": True,
            "formats": ["json", "md"],
            "families": [
                {
                    "name": "pomdp",
                    "status": "pass",
                    "model_count": 1,
                    "failed_models": [],
                    "models": [
                        {
                            "source_file": "model.md",
                            "round_trips": [
                                {
                                    "format": "json",
                                    "status": "pass",
                                    "differences": [],
                                    "artifact": "out.json",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        md = render_semantic_fidelity_markdown(ledger)
        assert "# GNN Semantic Fidelity Ledger" in md
        assert "pomdp" in md and "model.md" in md
        assert "Round Trips" in md

    @pytest.mark.unit
    def test_renders_failure_reason(self) -> None:
        """A failed round trip should surface its reason."""
        ledger: dict[str, Any] = {
            "schema": "v1",
            "family_count": 1,
            "strict": False,
            "families": [
                {
                    "name": "fam",
                    "status": "fail",
                    "model_count": 1,
                    "failed_models": ["m.md"],
                    "models": [
                        {
                            "source_file": "m.md",
                            "round_trips": [
                                {
                                    "format": "md",
                                    "status": "fail",
                                    "reason": "schema mismatch",
                                    "differences": [{"field": "A"}],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        md = render_semantic_fidelity_markdown(ledger)
        assert "schema mismatch" in md


class TestCrossFrameworkReliabilityLedger:
    """Render the cross-framework reliability ledger to Markdown."""

    @pytest.mark.unit
    def test_renders_framework_profiles(self) -> None:
        """Populated ledgers should include per-framework profile rows."""
        ledger: dict[str, Any] = {
            "schema": "v2-rx",
            "family_count": 1,
            "strict": True,
            "frameworks": ["rxinfer", "pymdp"],
            "families": [
                {
                    "name": "pomdp",
                    "status": "pass",
                    "comparison": {
                        "status": "pass",
                        "compared_frameworks": ["rxinfer"],
                    },
                    "required_framework_failures": [],
                    "frameworks": {
                        "rxinfer": {
                            "profile": "stable",
                            "status": "pass",
                            "reason": "ok",
                            "metrics": {"available": True},
                        }
                    },
                }
            ],
        }
        md = render_cross_framework_reliability_markdown(ledger)
        assert "# GNN Cross-Framework Reliability Ledger" in md
        assert "rxinfer" in md
        assert "Framework Profiles" in md

    @pytest.mark.unit
    def test_renders_missing_metrics_as_missing(self) -> None:
        """Frameworks without metrics should be labelled missing."""
        ledger: dict[str, Any] = {
            "schema": "v2-rl",
            "family_count": 1,
            "strict": False,
            "frameworks": ["jax"],
            "families": [
                {
                    "name": "fam",
                    "status": "fail",
                    "comparison": {"status": "fail", "compared_frameworks": []},
                    "required_framework_failures": ["jax"],
                    "frameworks": {
                        "jax": {
                            "profile": "unstable",
                            "status": "fail",
                            "reason": "no trace",
                            "metrics": {},
                        }
                    },
                }
            ],
        }
        md = render_cross_framework_reliability_markdown(ledger)
        assert "missing" in md