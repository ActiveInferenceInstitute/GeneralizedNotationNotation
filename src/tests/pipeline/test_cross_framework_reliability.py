from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.cross_framework_reliability import (
    compare_framework_metrics,
    run_cross_framework_reliability,
)


def test_cross_framework_gate_profiles_all_maintained_backends(
    tmp_path: Path,
) -> None:
    def acceptance_runner(_family: object, output_dir: Path) -> dict[str, Any]:
        pipeline_output = output_dir / "basics" / "pipeline_output"
        _write_simulation_payload(pipeline_output, "pymdp")
        _write_execution_summary(pipeline_output, "pymdp")
        return _acceptance_ledger("basics", pipeline_output)

    ledger = run_cross_framework_reliability(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics"],
        strict=True,
        acceptance_runner=acceptance_runner,
    )

    family = ledger["families"][0]
    assert ledger["status"] == "passed"
    assert family["frameworks"]["pymdp"]["status"] == "passed"
    assert family["frameworks"]["rxinfer"]["status"] == "unsupported"
    assert family["comparison"]["status"] == "skipped"
    assert (tmp_path / "cross_framework_reliability_ledger.json").exists()
    assert (tmp_path / "cross_framework_reliability_ledger.md").exists()


def test_cross_framework_gate_fails_missing_step_12_evidence(
    tmp_path: Path,
) -> None:
    def acceptance_runner(_family: object, output_dir: Path) -> dict[str, Any]:
        pipeline_output = output_dir / "basics" / "pipeline_output"
        _write_simulation_payload(pipeline_output, "pymdp")
        _write_execution_summary(pipeline_output, "pymdp")
        ledger = _acceptance_ledger("basics", pipeline_output)
        ledger["families"][0]["step_evidence"]["12"]["status"] = "failed"
        return ledger

    with pytest.raises(RuntimeError, match="Cross-framework reliability failed"):
        run_cross_framework_reliability(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            strict=True,
            acceptance_runner=acceptance_runner,
        )


def test_cross_framework_gate_fails_unprofiled_framework(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unprofiled frameworks"):
        run_cross_framework_reliability(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            frameworks=("pymdp", "made_up_backend"),
            strict=True,
        )


def test_compare_framework_metrics_fails_on_metric_mismatch() -> None:
    issues = compare_framework_metrics(
        {
            "pymdp": {
                "available": True,
                "random_seed": 42,
                "num_timesteps": 10,
                "matrix_shapes": {"B_shape": [3, 3, 2]},
                "trace_lengths": {"actions": 10},
            },
            "jax": {
                "available": True,
                "random_seed": 42,
                "num_timesteps": 11,
                "matrix_shapes": {"B_shape": [3, 3, 2]},
                "trace_lengths": {"actions": 10},
            },
        }
    )

    assert any(issue.field == "num_timesteps" for issue in issues)


def test_compare_framework_metrics_fails_missing_seed() -> None:
    issues = compare_framework_metrics(
        {
            "pymdp": {
                "available": True,
                "random_seed": 42,
                "num_timesteps": 10,
                "matrix_shapes": {},
                "trace_lengths": {},
            },
            "jax": {
                "available": True,
                "random_seed": None,
                "num_timesteps": 10,
                "matrix_shapes": {},
                "trace_lengths": {},
            },
        }
    )

    assert any(issue.field == "random_seed" for issue in issues)


def _acceptance_ledger(name: str, pipeline_output: Path) -> dict[str, Any]:
    return {
        "schema": "gnn_model_family_acceptance_ledger_v1",
        "status": "passed",
        "families": [
            {
                "name": name,
                "status": "passed",
                "command": [
                    "python",
                    "src/main.py",
                    "--output-dir",
                    str(pipeline_output),
                ],
                "step_evidence": {
                    "11": {"status": "passed"},
                    "12": {"status": "passed"},
                },
                "artifact_links": [str(pipeline_output / "12_execute_output")],
            }
        ],
    }


def _write_simulation_payload(pipeline_output: Path, framework: str) -> None:
    payload_dir = (
        pipeline_output / "12_execute_output" / "demo" / framework / "simulation_data"
    )
    payload_dir.mkdir(parents=True, exist_ok=True)
    (payload_dir / "demo_simulation_results.json").write_text(
        json.dumps(
            {
                "schema_version": f"{framework}_simulation_v1",
                "success": True,
                "random_seed": 42,
                "num_timesteps": 5,
                "observations": [0, 1, 0, 1, 0],
                "actions": [0, 0, 1, 1, 0],
                "beliefs": [[0.5, 0.5]] * 5,
                "model_parameters": {"B_shape": [2, 2, 2]},
            }
        ),
        encoding="utf-8",
    )


def _write_execution_summary(pipeline_output: Path, framework: str) -> None:
    result_path = (
        pipeline_output
        / "12_execute_output"
        / "demo"
        / framework
        / "execution_logs"
        / f"demo_{framework}_results.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"success": True}), encoding="utf-8")
    summary_path = (
        pipeline_output / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "execution_details": [
                    {
                        "framework": framework,
                        "script_name": f"demo_{framework}.py",
                        "success": True,
                        "skipped": False,
                        "return_code": 0,
                        "structured_result_file": str(result_path),
                        "implementation_directory": str(result_path.parents[1]),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
