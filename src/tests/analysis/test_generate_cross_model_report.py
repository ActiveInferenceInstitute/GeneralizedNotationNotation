"""Tests for the cross-model comparison report generator.

Covers ``analysis.generate_cross_model_report`` producing a valid markdown
report from sample analysis data and gracefully handling empty / missing /
unscoped inputs.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from analysis.generate_cross_model_report import (
    _action_diversity,
    _collect_execution_times,
    _collect_simulation_data,
    _mean_belief_confidence,
    _mean_belief_entropy,
    _mean_efe,
    _timestep_count,
    _validation_status,
    generate_cross_model_report,
)


def _sim_payload(
    beliefs: list[list[float]],
    actions: list[int],
    efe: list[list[float]],
    validation: dict[str, bool] | None = None,
    num_timesteps: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "beliefs": beliefs,
        "actions": actions,
        "efe_history": efe,
    }
    if num_timesteps is not None:
        payload["num_timesteps"] = num_timesteps
    if validation is not None:
        payload["validation"] = validation
    return payload


def _write_sim(
    exec_dir: Path,
    model_name: str,
    framework: str,
    payload: dict[str, object],
) -> Path:
    sim_dir = exec_dir / model_name / framework / "simulation_data"
    sim_dir.mkdir(parents=True)
    sim_file = sim_dir / "simulation_results.json"
    sim_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sim_file


def _write_times(exec_dir: Path, rows: list[dict[str, Any]]) -> None:
    summ = exec_dir / "summaries"
    summ.mkdir(parents=True)
    (summ / "execution_summary.json").write_text(
        json.dumps({"execution_results": rows}), encoding="utf-8"
    )


def _realistic_report_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 12_execute_output tree that `generate_cross_model_report` reads."""
    exec_dir = tmp_path / "12_execute_output"
    valid = {"all_valid": True}
    _write_sim(
        exec_dir,
        "gridworld",
        "pymdp",
        _sim_payload(
            [[0.8, 0.2, 0.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.0]],
            [1, 2, 1],
            [[1.0, 0.5], [0.8, 0.4], [0.6, 0.3]],
            valid,
            3,
        ),
    )
    _write_sim(
        exec_dir,
        "gridworld",
        "pytorch",
        _sim_payload(
            [[0.7, 0.3, 0.0], [0.8, 0.2, 0.0]],
            [0, 1],
            [[0.9, 0.4], [0.7, 0.1]],
            valid,
            2,
        ),
    )
    _write_sim(  # a model with no validation dict -> renders a dash
        exec_dir,
        "raw",
        "rxinfer",
        _sim_payload(
            [[0.6, 0.4], [0.5, 0.5]],
            [0, 0],
            [[0.2], [0.1]],
            None,
            2,
        ),
    )
    _write_times(
        exec_dir,
        [
            {
                "script": str(exec_dir / "gridworld" / "pymdp" / "sim.py"),
                "execution_time": 0.5,
            },
            {
                "script": str(exec_dir / "gridworld" / "pytorch" / "sim.py"),
                "execution_time": 1.2,
            },
            {
                "script": str(exec_dir / "raw" / "rxinfer" / "sim.jl"),
                "execution_time": 3.1,
            },
        ],
    )
    return exec_dir, tmp_path / "16_analysis_output"


class TestGenerateCrossModelReport:
    """End-to-end report generation from sample analysis data."""

    def test_report_written_with_full_structure(self, tmp_path: Path) -> None:
        exec_dir, analysis_dir = _realistic_report_inputs(tmp_path)
        analysis_dir.mkdir(parents=True)
        output_path = analysis_dir / "cross_model_comparison_report.md"

        result = generate_cross_model_report(exec_dir, analysis_dir, output_path)
        assert result == str(output_path)
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        assert "# Cross-Model Comparison Report" in content
        assert "## Summary Matrix" in content
        assert "## Expected Free Energy Comparison" in content
        assert "## Belief Entropy Comparison" in content
        assert "## Execution Time (seconds)" in content
        assert "## Per-Model Details" in content
        assert "## Cross-Model Observations" in content

        # Both models and frameworks with data appear.
        assert "**gridworld**" in content
        assert "**raw**" in content
        assert "PyMDP" in content
        assert "PyTorch" in content
        assert "RxInfer" in content

        # Pinned metric rendering: gridworld/pymdp mean confidence 0.9 and
        # the mean selected-action EFE 0.5333 across its 3 steps.
        assert "✅ 0.900" in content
        assert " 0.5333 |" in content
        # Execution times surface from the summary file.
        assert "0.50" in content
        assert "1.20" in content

    def test_report_empty_execution_dir_returns_empty(self, tmp_path: Path) -> None:
        exec_dir = tmp_path / "12_execute_output"
        exec_dir.mkdir(parents=True)
        analysis_dir = tmp_path / "16_analysis_output"
        analysis_dir.mkdir(parents=True)

        result = generate_cross_model_report(
            exec_dir, analysis_dir, analysis_dir / "report.md"
        )
        assert result == ""
        assert not (analysis_dir / "report.md").exists()

    def test_report_missing_execution_dir_returns_empty(self, tmp_path: Path) -> None:
        analysis_dir = tmp_path / "16_analysis_output"
        analysis_dir.mkdir(parents=True)
        result = generate_cross_model_report(
            tmp_path / "missing", analysis_dir, analysis_dir / "report.md"
        )
        assert result == ""

    def test_report_allowed_scope_filters_models_and_frameworks(
        self, tmp_path: Path
    ) -> None:
        exec_dir, analysis_dir = _realistic_report_inputs(tmp_path)
        analysis_dir.mkdir(parents=True)
        output_path = analysis_dir / "scoped_report.md"

        result = generate_cross_model_report(
            exec_dir,
            analysis_dir,
            output_path,
            allowed_frameworks={"pytorch"},
            allowed_model_names={"gridworld"},
        )
        assert result == str(output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "PyTorch" in content
        assert "PyMDP" not in content
        assert "RxInfer" not in content
        assert "**raw**" not in content

    def test_report_metric_extractors_on_nested_schema(self) -> None:
        """Metric helpers understand beliefs/efe behind simulation_trace."""
        result = {
            "simulation_trace": {
                "beliefs": [[0.6, 0.4], [0.8, 0.2]],
                "actions": [0, 1],
                "efe_history": [[0.3], [0.1]],
            }
        }
        assert _mean_belief_confidence(result) == pytest.approx(0.7)
        assert _mean_belief_entropy(result) is not None
        assert _mean_efe(result) == pytest.approx(0.2)
        assert _action_diversity(result) == pytest.approx(1.0)
        assert _timestep_count(result) == 2

    def test_metric_helpers_graceful_on_empty(self) -> None:
        assert _mean_belief_confidence({}) is None
        assert _mean_belief_entropy({}) is None
        assert _mean_efe({}) is None
        assert _action_diversity({}) is None
        assert _timestep_count({}) is None
        assert _validation_status({}) == "—"

    def test_validation_status_combinations(self) -> None:
        assert _validation_status({"validation": {"a": True, "b": True}}) == "✅"
        assert _validation_status({"validation": {"a": True, "b": False}}) == "❌"
        assert _validation_status({"validation": {}}) == "—"


class TestCollectHelpers:
    """Direct checks on the private collection helpers."""

    def test_collect_simulation_data_malformed_file_skipped(
        self, tmp_path: Path
    ) -> None:
        sim_dir = tmp_path / "m" / "pymdp" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            "{ bad", encoding="utf-8"
        )
        data = _collect_simulation_data(tmp_path)
        assert data == {}

    def test_collect_execution_times_nested_summaries(self, tmp_path: Path) -> None:
        script = str(tmp_path / "m" / "pymdp" / "sim.py")
        _write_times(tmp_path, [{"script": script, "execution_time": 2.5}])
        times = _collect_execution_times(tmp_path)
        assert times.get("m", {}).get("pymdp") == pytest.approx(2.5)

    def test_collect_execution_times_missing_summary(self, tmp_path: Path) -> None:
        times = _collect_execution_times(tmp_path)
        assert times == {}