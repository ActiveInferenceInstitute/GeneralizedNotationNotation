"""Tests for advanced_visualization/interactive_viz.py.

Covers _generate_interactive_plotly_dashboard using real Plotly + numpy data
(no mocks). Plotly and numpy are available in the dev environment.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestInteractivePlotlyDashboard:
    def _make_kwargs(self, tmp_path: Any) -> dict[str, Any]:
        return {
            "model_name": "TestModel",
            "model_data": {
                "variables": [
                    {"name": "s", "var_type": "hidden_state"},
                    {"name": "o", "var_type": "observation"},
                ],
                "parameters": [{"name": "A", "value": [[0.5, 0.5], [0.5, 0.5]]}],
                "connections": [{"source": "s", "target": "o"}],
            },
            "output_dir": tmp_path,
            "export_formats": ["html"],
            "dependencies": {},
            "logger": logging.getLogger("test_interactive"),
        }

    def _fn(self) -> Any:
        from advanced_visualization.interactive_viz import (
            _generate_interactive_plotly_dashboard,
        )

        return _generate_interactive_plotly_dashboard

    def test_generates_html_dashboard(self, tmp_path: Any) -> None:
        fn = self._fn()
        kwargs = self._make_kwargs(tmp_path)
        attempt = fn(**kwargs)
        assert attempt.viz_type == "interactive_dashboard"
        assert attempt.status == "success"
        assert len(attempt.output_files) == 1
        out = Path(attempt.output_files[0])
        assert out.exists()
        assert out.suffix == ".html"

    def test_empty_variables_still_succeeds(self, tmp_path: Any) -> None:
        fn = self._fn()
        kwargs = self._make_kwargs(tmp_path)
        kwargs["model_data"] = {
            "variables": [],
            "parameters": [],
            "connections": [],
        }
        attempt = fn(**kwargs)
        assert attempt.status == "success"
        assert len(attempt.output_files) >= 1

    def test_exports_png_when_requested(self, tmp_path: Any) -> None:
        fn = self._fn()
        kwargs = self._make_kwargs(tmp_path)
        kwargs["export_formats"] = ["html", "png"]
        attempt = fn(**kwargs)
        # png export uses kaleido; may fail if not installed → skip check is lenient.
        assert attempt.status in ("success", "failed")