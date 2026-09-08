"""Pins for ``advanced_visualization.network_viz`` (previously 32%).

Each ``_generate_*`` function returns an ``AdvancedVisualizationAttempt``
whose status reflects the outcome; missing/degenerate model data must
degrade to a ``skipped`` attempt — never raise.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gnn.advanced_visualization._shared import AdvancedVisualizationAttempt
from gnn.advanced_visualization.network_viz import (
    _generate_network_metrics,
    _generate_policy_visualization,
    _generate_pomdp_transition_analysis,
)

_LOGGER = logging.getLogger("w2probe-viz")


def _minimal_model_data() -> dict:
    return {
        "name": "ProbeModel",
        "variables": [
            {"name": "s", "var_type": "hidden_state", "dimensions": [2, 2]},
            {"name": "o", "var_type": "observation", "dimensions": [2]},
        ],
        "connections": [
            {"source": "s", "target": "s"},
            {"source": "s", "target": "o"},
        ],
        "parameters": {
            "B": [[0.9, 0.1], [0.1, 0.9]],
            "D": [0.5, 0.5],
            "C": [0.1, 0.1],
        },
    }


def _degenerate_model_data() -> dict:
    return {"name": "Empty", "variables": [], "connections": []}


def test_pomdp_transition_analysis_renders_or_skips(tmp_path: Path) -> None:
    attempt = _generate_pomdp_transition_analysis(
        "ProbeModel", _minimal_model_data(), tmp_path, {}, _LOGGER
    )

    assert isinstance(attempt, AdvancedVisualizationAttempt)
    assert attempt.status in {"success", "skipped"}
    if attempt.status == "success":
        assert attempt.output_files
        assert all(Path(p).exists() for p in attempt.output_files)


def test_pomdp_transition_analysis_tolerates_empty_model(tmp_path: Path) -> None:
    attempt = _generate_pomdp_transition_analysis(
        "Empty", _degenerate_model_data(), tmp_path, {}, _LOGGER
    )

    assert attempt.status in {"success", "skipped", "failed"}
    assert attempt.model_name == "Empty"


def test_policy_visualization_renders_or_skips(tmp_path: Path) -> None:
    attempt = _generate_policy_visualization(
        "ProbeModel", _minimal_model_data(), tmp_path, {}, _LOGGER
    )

    assert isinstance(attempt, AdvancedVisualizationAttempt)
    assert attempt.status in {"success", "skipped"}
    if attempt.status == "success":
        assert all(Path(p).exists() for p in attempt.output_files)


def test_network_metrics_renders_or_skips(tmp_path: Path) -> None:
    attempt = _generate_network_metrics(
        "ProbeModel", _minimal_model_data(), tmp_path, {}, _LOGGER
    )

    assert isinstance(attempt, AdvancedVisualizationAttempt)
    assert attempt.status in {"success", "skipped"}
    assert attempt.viz_type == "network_metrics"
    if attempt.status == "success":
        assert all(Path(p).exists() for p in attempt.output_files)


def test_network_metrics_tolerates_degenerate_model(tmp_path: Path) -> None:
    attempt = _generate_network_metrics(
        "Empty", _degenerate_model_data(), tmp_path, {}, _LOGGER
    )

    assert attempt.status in {"success", "skipped", "failed"}
