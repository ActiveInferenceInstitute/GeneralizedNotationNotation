"""Tests for the shared flat-payload analyzer (PyTorch/NumPyro dedup).

Pins the behavior of ``analysis.flat_payload_analyzer``:
- ``compute_flat_payload_metrics`` pure metric computation
- ``discover_result_files`` file discovery + root recovery
- ``generate_analysis_from_logs`` end-to-end via the PyTorch spec
- graceful degradation when matplotlib is unavailable
- framework isolation (one analyzer ignores the other's results)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.flat_payload_analyzer import (  # noqa: E402
    FlatPayloadSpec,
    compute_flat_payload_metrics,
    discover_result_files,
    generate_analysis_from_logs,
)


def _realistic_payload(model_name: str = "model_a") -> dict[str, Any]:
    return {
        "model_name": model_name,
        "beliefs": [[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]],
        "actions": [1, 2, 1],
        "observations": [0, 1, 0],
        "efe_history": [[0.5, 0.6], [0.4, 0.5], [0.3, 0.8]],
        "validation": {"all_valid": True},
    }


_TEST_SPEC = FlatPayloadSpec(
    framework="testfw",
    file_patterns=(
        "**/testfw/**/simulation_results.json",
        "**/testfw_simulation_results.json",
    ),
    analysis_filename="testfw_analysis.json",
    title_prefix="TestFW",
    bar_color="#FF0000",
    log_label="TestFW",
)


class TestComputeFlatPayloadMetrics:
    """Pin the pure metric-computation contract."""

    @pytest.mark.unit
    def test_metrics_from_realistic_payload(self) -> None:
        beliefs = np.array([[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]])
        actions: list[Any] = [1, 2, 1]
        efe = np.array([[0.5, 0.6], [0.4, 0.5], [0.3, 0.8]])
        metrics = compute_flat_payload_metrics(beliefs, actions, efe)
        assert metrics["mean_confidence"] == pytest.approx(0.8, abs=0.01)
        assert metrics["final_confidence"] == pytest.approx(0.9, abs=0.01)
        assert metrics["action_distribution"] == {1: 2, 2: 1}
        assert metrics["mean_efe"] == pytest.approx(0.5166, abs=0.01)
        assert "mean_belief_entropy" in metrics
        assert "final_belief_entropy" in metrics

    @pytest.mark.unit
    def test_empty_beliefs_yields_no_entropy(self) -> None:
        beliefs = np.array([]).reshape(0, 0)
        metrics = compute_flat_payload_metrics(beliefs, [], np.array([]).reshape(0, 0))
        assert "mean_belief_entropy" not in metrics
        assert "mean_confidence" not in metrics
        assert "mean_efe" not in metrics

    @pytest.mark.unit
    def test_beliefs_without_actions(self) -> None:
        beliefs = np.array([[0.5, 0.5], [0.6, 0.4]])
        metrics = compute_flat_payload_metrics(beliefs, [], np.array([]).reshape(0, 0))
        assert "mean_confidence" in metrics
        assert "action_distribution" not in metrics

    @pytest.mark.unit
    def test_1d_beliefs_skipped(self) -> None:
        beliefs = np.array([0.5, 0.5])
        metrics = compute_flat_payload_metrics(beliefs, [], np.array([]).reshape(0, 0))
        assert "mean_confidence" not in metrics


class TestDiscoverResultFiles:
    """Pin the file-discovery contract."""

    @pytest.mark.unit
    def test_nested_pattern_match(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "testfw" / "simulation_data"
        sim_dir.mkdir(parents=True)
        sim_file = sim_dir / "simulation_results.json"
        sim_file.write_text(json.dumps(_realistic_payload()))
        found = discover_result_files(tmp_path, _TEST_SPEC)
        assert sim_file in found

    @pytest.mark.unit
    def test_prefixed_pattern_match(self, tmp_path: Path) -> None:
        sim_file = tmp_path / "testfw_simulation_results.json"
        sim_file.write_text(json.dumps(_realistic_payload()))
        found = discover_result_files(tmp_path, _TEST_SPEC)
        assert sim_file in found

    @pytest.mark.unit
    def test_root_recovery_fallback(self, tmp_path: Path) -> None:
        sim_file = tmp_path / "simulation_results.json"
        sim_file.write_text(json.dumps(_realistic_payload()))
        found = discover_result_files(tmp_path, _TEST_SPEC)
        assert sim_file in found

    @pytest.mark.unit
    def test_no_files_returns_empty(self, tmp_path: Path) -> None:
        assert discover_result_files(tmp_path, _TEST_SPEC) == []


class TestGenerateAnalysisFromLogs:
    """End-to-end shared-analyzer contract via the TestFW spec."""

    @pytest.mark.unit
    def test_end_to_end_writes_analysis_json(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "testfw" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(_realistic_payload())
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        generated = generate_analysis_from_logs(_TEST_SPEC, tmp_path, out_dir)
        assert len(generated) == 1
        analysis_file = Path(generated[0])
        assert analysis_file.exists()
        analysis = json.loads(analysis_file.read_text())
        assert analysis["framework"] == "testfw"
        assert analysis["model_name"] == "model_a"
        assert analysis["num_timesteps"] == 3
        assert analysis["validation"] == {"all_valid": True}
        assert analysis["metrics"]["action_distribution"] == {"1": 2, "2": 1}

    @pytest.mark.unit
    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert (
            generate_analysis_from_logs(_TEST_SPEC, tmp_path / "nonexistent", out_dir)
            == []
        )

    @pytest.mark.unit
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert generate_analysis_from_logs(_TEST_SPEC, tmp_path, out_dir) == []

    @pytest.mark.unit
    def test_default_output_dir_defaults_to_results_dir(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "testfw" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(_realistic_payload())
        )
        generated = generate_analysis_from_logs(_TEST_SPEC, tmp_path)
        assert len(generated) == 1
        # Default output_dir = results_dir → analysis under tmp_path/model_a/
        analysis_path = tmp_path / "model_a" / "testfw_analysis.json"
        assert analysis_path.exists()

    @pytest.mark.unit
    def test_framework_isolation_ignores_other_framework(self, tmp_path: Path) -> None:
        # Write a "otherfw" result that should be ignored.
        other_dir = tmp_path / "model_a" / "otherfw" / "simulation_data"
        other_dir.mkdir(parents=True)
        (other_dir / "simulation_results.json").write_text(
            json.dumps(_realistic_payload())
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert generate_analysis_from_logs(_TEST_SPEC, tmp_path, out_dir) == []

    @pytest.mark.unit
    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        sim_dir = tmp_path / "model_a" / "testfw" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text("{ not valid json }")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        assert generate_analysis_from_logs(_TEST_SPEC, tmp_path, out_dir) == []

    @pytest.mark.unit
    def test_graceful_degradation_without_matplotlib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sim_dir = tmp_path / "model_a" / "testfw" / "simulation_data"
        sim_dir.mkdir(parents=True)
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(_realistic_payload())
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        generated = generate_analysis_from_logs(_TEST_SPEC, tmp_path, out_dir)
        assert len(generated) == 1
        analysis = json.loads(Path(generated[0]).read_text())
        assert analysis["plots_generated"] is False


class TestFlatPayloadSpec:
    """The spec dataclass is frozen and fields are accessible."""

    @pytest.mark.unit
    def test_spec_is_frozen(self) -> None:
        spec = FlatPayloadSpec(
            framework="x",
            file_patterns=("a", "b"),
            analysis_filename="x.json",
            title_prefix="X",
            bar_color="#000",
            log_label="X",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.framework = "y"  # type: ignore[misc]

    @pytest.mark.unit
    def test_spec_fields_accessible(self) -> None:
        assert _TEST_SPEC.framework == "testfw"
        assert _TEST_SPEC.analysis_filename == "testfw_analysis.json"
        assert _TEST_SPEC.bar_color == "#FF0000"
