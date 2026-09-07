"""Real-behavior tests for ``src/gnn/pipeline/diagnostic_enhancer.py``.

The enhancer is the module that turns a raw pipeline execution summary into
health scores, dependency analysis, and recommendations, and it writes the
``enhanced_<name>.json`` artifact beside the input. Before this file it had
no direct tests, so its contracts — including the fail-silent ``{}`` return
on unreadable input — were unpinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gnn.pipeline.diagnostic_enhancer import PipelineDiagnosticEnhancer


def _step(
    script_name: str,
    status: str,
    duration: float = 1.0,
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "script_name": script_name,
        "status": status,
        "duration_seconds": duration,
        "stderr": stderr,
        "stdout": "",
    }


def _write_summary(tmp_path: Path, steps: list[dict[str, Any]]) -> Path:
    summary_path = tmp_path / "pipeline_execution_summary.json"
    summary_path.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    return summary_path


def test_failed_step_annotates_diagnostics_and_writes_enhanced_file(
    tmp_path: Path,
) -> None:
    summary_path = _write_summary(
        tmp_path,
        [
            _step("3_gnn.py", "SUCCESS", duration=2.0),
            _step(
                "11_render.py",
                "FAILED",
                duration=4.0,
                stderr="ModuleNotFoundError: No module named 'pymdp'",
            ),
        ],
    )

    enhanced = PipelineDiagnosticEnhancer().enhance_summary(summary_path)

    assert enhanced, "enhance_summary must return the annotated summary"
    execution = enhanced["diagnostics"]["execution_analysis"]
    assert execution["total_steps"] == 2
    assert execution["failed_steps"] == 1
    assert execution["successful_steps"] == 1
    assert execution["success_rate"] == 50.0

    deps = enhanced["diagnostics"]["dependency_analysis"]
    assert deps["missing_dependencies"] == ["pymdp"]

    rec_types = {rec["type"] for rec in enhanced["recommendations"]}
    assert "critical" in rec_types
    assert "dependency" in rec_types

    # 100 - 25 (one failed step) = 75 -> "good"
    assert enhanced["health_score"]["score"] == 75
    assert enhanced["health_score"]["rating"] == "good"

    enhanced_path = tmp_path / "enhanced_pipeline_execution_summary.json"
    assert enhanced_path.is_file(), "enhanced artifact must be written beside input"
    assert json.loads(enhanced_path.read_text(encoding="utf-8")) == enhanced


def test_healthy_pipeline_scores_excellent_with_no_recommendations(
    tmp_path: Path,
) -> None:
    summary_path = _write_summary(
        tmp_path,
        [
            _step("3_gnn.py", "SUCCESS", duration=0.5),
            _step("11_render.py", "SUCCESS", duration=1.5),
        ],
    )

    enhanced = PipelineDiagnosticEnhancer().enhance_summary(summary_path)

    assert enhanced["health_score"] == {
        "score": 100,
        "rating": "excellent",
        "deductions": 0,
        "breakdown": {
            "base_score": 100,
            "failure_penalty": 0,
            "warning_penalty": 0,
            "performance_penalty": 0,
        },
    }
    assert enhanced["recommendations"] == []


def test_warning_step_and_optional_dependency_are_reported(tmp_path: Path) -> None:
    summary_path = _write_summary(
        tmp_path,
        [
            _step("3_gnn.py", "SUCCESS", duration=1.0),
            _step(
                "12_execute.py",
                "SUCCESS_WITH_WARNINGS",
                duration=1.0,
                stderr="julia not available; skipping RxInfer execution",
            ),
        ],
    )

    enhanced = PipelineDiagnosticEnhancer().enhance_summary(summary_path)

    execution = enhanced["diagnostics"]["execution_analysis"]
    assert execution["warning_steps"] == 1
    assert execution["failed_steps"] == 0

    deps = enhanced["diagnostics"]["dependency_analysis"]
    assert deps["optional_dependencies"] == ["julia"]

    # 100 - 5 (one warning step) = 95 -> "excellent"
    assert enhanced["health_score"]["score"] == 95
    assert enhanced["health_score"]["rating"] == "excellent"


def test_slow_average_duration_triggers_optimization_recommendation(
    tmp_path: Path,
) -> None:
    summary_path = _write_summary(
        tmp_path,
        [
            _step("8_visualization.py", "SUCCESS", duration=8.0),
            _step("16_analysis.py", "SUCCESS", duration=6.0),
        ],
    )

    enhanced = PipelineDiagnosticEnhancer().enhance_summary(summary_path)

    optimization = [
        rec
        for rec in enhanced["recommendations"]
        if rec["type"] == "optimization" and rec["category"] == "performance"
    ]
    assert optimization, "average duration 7.0s must trigger the optimization rec"
    assert optimization[0]["description"].startswith("Step 8_visualization.py")

    # avg 7.0s <= 10.0s, so no health penalty beyond the recommendation
    assert enhanced["health_score"]["deductions"] == 0


def test_missing_summary_file_returns_empty_dict(tmp_path: Path) -> None:
    """NEGATIVE: unreadable input fails silent — the enhancer returns ``{}``.

    This pins the current fail-silent contract (the error is logged, nothing
    raises, no file is written). If the module is later changed to propagate
    the failure, this test is the one that flips deliberately.
    """
    missing = tmp_path / "does_not_exist.json"

    assert PipelineDiagnosticEnhancer().enhance_summary(missing) == {}
    assert not (tmp_path / "enhanced_does_not_exist.json").exists()


def test_malformed_summary_json_returns_empty_dict(tmp_path: Path) -> None:
    """NEGATIVE: malformed JSON fails silent — the enhancer returns ``{}``.

    Documents the same fail-silent contract as the missing-file case: a
    corrupted summary must not produce a half-annotated enhanced artifact.
    """
    bad = tmp_path / "pipeline_execution_summary.json"
    bad.write_text("{not json", encoding="utf-8")

    assert PipelineDiagnosticEnhancer().enhance_summary(bad) == {}
    assert not (tmp_path / "enhanced_pipeline_execution_summary.json").exists()
