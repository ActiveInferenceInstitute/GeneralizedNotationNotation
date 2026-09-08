"""Pins for ``analysis/post_simulation.analyze_execution_results`` (8% coverage).

Builds a synthetic execution-output tree and asserts the documented grouping,
normalization, and filtering contracts of the Step-16 entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gnn.analysis.post_simulation import analyze_execution_results

_PSYMDP_PAYLOAD: dict[str, Any] = {
    "framework": "PyMDP",
    "model_name": "Simple MDP Agent",
    "schema_version": "pymdp_simulation_v1",
    "beliefs_by_factor": {"joint_state": [0.9, 0.1]},
    "observations_by_modality": {"joint_observation": [1, 0]},
    "actions_by_control_factor": {"joint_action": [0]},
    "hidden_states_by_factor": {"joint_state": [1, 0]},
    "expected_free_energy": [1.5, 1.2],
    "policy_posterior": [0.7, 0.3],
}


def _jax_payload(model_name: str = "Simple MDP Agent") -> dict[str, Any]:
    return {
        "framework": "JAX",
        "model_name": model_name,
        "schema_version": "jax_simulation_v1",
        "beliefs": [0.8, 0.2],
        "observations": [0, 1],
        "actions": [1],
        "expected_free_energy": [2.0, 1.9],
        "policy_posterior": [0.6, 0.4],
    }


def _write_results(output: Path, payloads: list[dict[str, Any]]) -> None:
    for index, payload in enumerate(payloads):
        target = output / f"run{index}_results.json"
        target.write_text(json.dumps(payload), encoding="utf-8")


def test_empty_dir_yields_empty_framework_results(tmp_path: Path) -> None:
    result = analyze_execution_results(tmp_path)

    assert result["framework_results"] == {}
    assert result["cross_framework_comparison"] == {}
    assert result["execution_results_dir"] == str(tmp_path)


def test_groups_and_normalizes_framework_names(tmp_path: Path) -> None:
    _write_results(tmp_path, [_PSYMDP_PAYLOAD, _jax_payload()])

    result = analyze_execution_results(tmp_path)

    # "PyMDP"/"JAX" display names normalize to lowercase group keys.
    assert set(result["framework_results"]) == {"pymdp", "jax"}
    assert result["framework_results"]["pymdp"]["result_count"] == 1
    assert result["framework_results"]["jax"]["result_count"] == 1


def test_allowed_frameworks_filter(tmp_path: Path) -> None:
    _write_results(tmp_path, [_PSYMDP_PAYLOAD, _jax_payload()])

    result = analyze_execution_results(tmp_path, allowed_frameworks={"pymdp"})

    assert set(result["framework_results"]) == {"pymdp"}


def test_model_name_slug_filter_matches_display_names(tmp_path: Path) -> None:
    _write_results(tmp_path, [_PSYMDP_PAYLOAD, _jax_payload("Other Model")])

    result = analyze_execution_results(tmp_path, model_name="simple_mdp")

    assert result["framework_results"]["pymdp"]["result_count"] == 1
    assert "jax" not in result["framework_results"]


def test_malformed_result_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken_results.json").write_text("{not json", encoding="utf-8")
    _write_results(tmp_path, [_PSYMDP_PAYLOAD])

    result = analyze_execution_results(tmp_path)

    assert result["framework_results"]["pymdp"]["result_count"] == 1
