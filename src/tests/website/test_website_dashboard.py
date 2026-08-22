"""Tests for website/dashboard.py's self-contained HTML dashboard builder.

Covers render_dashboard and its internal helper functions using real
temporary filesystem content from the live website surface.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLoadJson:
    def test_missing_file_returns_empty_dict(self, tmp_path: Any) -> None:
        from website.dashboard import _load_json

        assert _load_json(tmp_path / "missing.json") == {}

    def test_invalid_json_returns_empty_dict(self, tmp_path: Any) -> None:
        from website.dashboard import _load_json

        p = tmp_path / "bad.json"
        p.write_text("{ not valid json")
        assert _load_json(p) == {}

    def test_valid_json_round_trips(self, tmp_path: Any) -> None:
        from website.dashboard import _load_json

        p = tmp_path / "good.json"
        p.write_text(json.dumps({"success": True, "steps": [{"name": "x"}]}))
        data = _load_json(p)
        assert data["success"] is True
        assert data["steps"][0]["name"] == "x"


class TestStatusColor:
    def _fn(self) -> Any:
        from website.dashboard import _status_color

        return _status_color

    def test_success_and_passed_green(self) -> None:
        assert self._fn()("success") == "#22c55e"
        assert self._fn()("passed") == "#22c55e"

    def test_warning_matches_warn(self) -> None:
        assert self._fn()("warning") == "#f59e0b"

    def test_failed_and_error_red(self) -> None:
        assert self._fn()("failed") == "#ef4444"
        assert self._fn()("error") == "#ef4444"

    def test_unknown_gray(self) -> None:
        assert self._fn()("running") == "#6b7280"
        assert self._fn()("") == "#6b7280"


class TestDiscoverStepDirs:
    def test_only_digit_output_dirs(self, tmp_path: Any) -> None:
        from website.dashboard import _discover_step_dirs

        (tmp_path / "00_foo_output").mkdir()
        (tmp_path / "01_bar_output").mkdir()
        (tmp_path / "assets").mkdir()
        (tmp_path / "not_digit_output").mkdir()
        result = _discover_step_dirs(tmp_path)
        names = [d.name for d in result]
        assert "00_foo_output" in names
        assert "01_bar_output" in names
        assert "assets" not in names
        assert "not_digit_output" not in names


class TestRenderHelpers:
    def test_render_sidebar_with_steps(self) -> None:
        from website.dashboard import _render_sidebar

        html = _render_sidebar([{"name": "Step 1", "status": "success"}], [])
        assert "Step 1" in html
        assert "#22c55e" in html

    def test_render_sidebar_falls_back_to_step_dirs(self, tmp_path: Any) -> None:
        from website.dashboard import _render_sidebar

        (tmp_path / "00_parse_output").mkdir()
        html = _render_sidebar([], [tmp_path / "00_parse_output"])
        assert "00_parse_output" in html

    def test_render_timeline_svg_no_steps(self) -> None:
        from website.dashboard import _render_timeline_svg

        html = _render_timeline_svg([])
        assert "No timing data" in html

    def test_render_timeline_svg_with_steps(self) -> None:
        from website.dashboard import _render_timeline_svg

        steps = [
            {"name": "Step A", "status": "success", "duration_seconds": 2.0},
            {"name": "Step B", "status": "failed", "duration_seconds": 5.0},
        ]
        html = _render_timeline_svg(steps)
        assert "<svg" in html
        assert "Step A" in html
        assert "Step B" in html

    def test_render_stats_with_step_dirs(self, tmp_path: Any) -> None:
        from website.dashboard import _render_stats

        d = tmp_path / "00_parse_output"
        d.mkdir()
        (d / "model.gnn").write_text("content")
        html = _render_stats({"success": True}, [d])
        assert "Steps" in html
        assert "Artifacts" in html


class TestRenderDashboard:
    def test_renders_dashboard_to_output(self, tmp_path: Any) -> None:
        from website.dashboard import render_dashboard

        results = tmp_path / "results"
        results.mkdir()
        # Empty summary → warning-only, still writes output.
        (results / "pipeline_execution_summary.json").write_text(
            json.dumps({"success": True, "steps": [], "total_duration": "N/A"})
        )
        out = tmp_path / "dashboard.html"
        assert render_dashboard(results, out) is True
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "GNN Pipeline Dashboard" in content

    def test_missing_summary_uses_defaults(self, tmp_path: Any) -> None:
        from website.dashboard import render_dashboard

        results = tmp_path / "results"
        results.mkdir()
        out = tmp_path / "dash.html"
        assert render_dashboard(results, out) is True
        assert out.exists()

    def test_badge_variants(self, tmp_path: Any) -> None:
        from website.dashboard import render_dashboard

        for success_val, badge_frag in [(True, "SUCCESS"), (False, "FAILED"), (None, "UNKNOWN")]:
            results = tmp_path / f"r_{success_val}"
            results.mkdir()
            summary: dict[str, object] = {"success": success_val, "steps": [], "total_duration": 42}
            (results / "pipeline_execution_summary.json").write_text(json.dumps(summary))
            out = tmp_path / f"out_{success_val}.html"
            assert render_dashboard(results, out) is True
            assert badge_frag in out.read_text()

    def test_summary_path_override(self, tmp_path: Any) -> None:
        from website.dashboard import render_dashboard

        results = tmp_path / "results"
        results.mkdir()
        summary = tmp_path / "custom_summary.json"
        summary.write_text(json.dumps({"steps": [{"name": "X", "status": "passed"}], "success": True}))
        out = tmp_path / "dash.html"
        assert render_dashboard(results, out, summary_path=summary) is True
        assert "X" in out.read_text()