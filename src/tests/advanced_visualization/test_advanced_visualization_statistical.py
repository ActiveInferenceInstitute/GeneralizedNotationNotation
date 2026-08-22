"""Tests for advanced_visualization/statistical_viz.py.

Covers _generate_statistical_plots and _generate_matrix_correlations using
real matplotlib + numpy data against the Agg backend.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStatisticalPlots:
    def _fn(self) -> Any:
        from advanced_visualization.statistical_viz import (
            _generate_statistical_plots,
        )

        return _generate_statistical_plots

    def _model_data(self) -> dict[str, Any]:
        return {
            "variables": [
                {"name": "s", "var_type": "hidden_state", "dimensions": [3, 1]},
                {"name": "o", "var_type": "observation", "dimensions": [2, 1]},
            ],
            "parameters": [
                {"name": "A", "value": [[0.5, 0.5], [0.5, 0.5]]},
                {"name": "alpha", "value": 0.7},
            ],
            "connections": [{"source": "s", "target": "o"}],
        }

    def _deps(self) -> dict[str, bool]:
        return {}

    def test_generates_statistical_png(self, tmp_path: Any) -> None:
        fn = self._fn()
        attempt = fn(
            "ModelA",
            self._model_data(),
            tmp_path,
            self._deps(),
            logging.getLogger("test_stat"),
        )
        assert attempt.status == "success"
        assert len(attempt.output_files) == 1
        out = Path(attempt.output_files[0])
        assert out.exists()
        assert out.suffix == ".png"

    def test_empty_variables_still_succeeds(self, tmp_path: Any) -> None:
        fn = self._fn()
        attempt = fn(
            "Empty",
            {"variables": [], "parameters": [], "connections": []},
            tmp_path,
            self._deps(),
            logging.getLogger("test_stat_empty"),
        )
        assert attempt.status == "success"


class TestMatrixCorrelations:
    def _fn(self) -> Any:
        from advanced_visualization.statistical_viz import (
            _generate_matrix_correlations,
        )

        return _generate_matrix_correlations

    def test_generates_correlation_plot_with_two_matrices(self, tmp_path: Any) -> None:
        fn = self._fn()
        model_data: dict[str, Any] = {
            "parameters": [
                {"name": "A", "value": [[0.5, 0.5], [0.5, 0.5]]},
                {"name": "B", "value": [[0.8, 0.2], [0.3, 0.7]]},
            ]
        }
        attempt = fn(
            "Corr", model_data, tmp_path, {}, logging.getLogger("test_corr")
        )
        assert attempt.status == "success"
        assert len(attempt.output_files) >= 1
        assert Path(attempt.output_files[0]).exists()

    def test_skips_with_single_matrix(self, tmp_path: Any) -> None:
        fn = self._fn()
        model_data: dict[str, Any] = {
            "parameters": [{"name": "A", "value": [[0.5, 0.5]]}]
        }
        attempt = fn(
            "Solo", model_data, tmp_path, {}, logging.getLogger("test_solo")
        )
        # Not enough matrices → skipped (warning-only), not a hard failure.
        assert attempt.status in ("skipped", "success")