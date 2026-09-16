#!/usr/bin/env python3
"""Cross-framework comparison degradation contracts.

One framework's failure/skip/render-failure status must degrade that
framework's column to an explicit status (vocabulary unified with the Step 12
execution summary) and never crash the comparison. All tests are offline:
the comparison input is a synthetic ``summaries/execution_summary.json``.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.analysis.framework_comparison import (  # noqa: E402
    analyze_framework_outputs,
    generate_framework_comparison_report,
)

LOGGER = logging.getLogger("test-framework-comparison-status")


def _write_execution_summary(
    execution_dir: Path,
    details: List[Dict[str, Any]],
    render_failures: Optional[List[Dict[str, str]]] = None,
) -> Path:
    summaries = execution_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    summary_file = summaries / "execution_summary.json"
    payload: Dict[str, Any] = {"execution_details": details}
    if render_failures is not None:
        payload["render_failures"] = render_failures
    summary_file.write_text(json.dumps(payload))
    return summary_file


def _summary(output: Dict[str, Any], framework: str) -> Dict[str, Any]:
    return output["frameworks"][framework]


# ── Explicit statuses for failed / skipped / healthy frameworks ───────────


def test_failed_skipped_and_healthy_frameworks_get_explicit_statuses(
    tmp_path: Path,
) -> None:
    _write_execution_summary(
        tmp_path,
        [
            {
                "framework": "pymdp",
                "success": True,
                "skipped": False,
                "execution_time": 1.0,
            },
            {
                "framework": "jax",
                "success": False,
                "skipped": True,
                "error": "Dependency not installed: jax",
            },
            {
                "framework": "discopy",
                "success": False,
                "skipped": False,
                "error": "Script failed with return code 2",
                "execution_time": 0.5,
            },
        ],
    )

    output = analyze_framework_outputs(tmp_path, LOGGER)

    pymdp = _summary(output, "pymdp")
    jax = _summary(output, "jax")
    discopy = _summary(output, "discopy")
    assert pymdp["status"] == "success"
    assert pymdp["status_reason"] is None
    assert pymdp["success_count"] == 1 and pymdp["total_count"] == 1
    assert jax["status"] == "skipped"
    assert jax["status_reason"] == "Dependency not installed: jax"
    assert jax["skipped_count"] == 1 and jax["success_count"] == 0
    assert discopy["status"] == "failed"
    assert discopy["status_reason"] == "Script failed with return code 2"
    assert discopy["failed_count"] == 1

    # The comparison still runs end to end on the degraded set.
    comparisons = output["comparisons"]
    assert comparisons["success_rates"] == {"pymdp": 1.0, "jax": 0.0, "discopy": 0.0}
    metrics = output["metrics"]
    assert metrics["total_skipped"] == 1
    assert metrics["total_failed"] == 1
    assert metrics["total_successful"] == 1
    assert metrics["total_executions"] == 3


def test_success_with_skips_status_unified_with_execute_summary(
    tmp_path: Path,
) -> None:
    _write_execution_summary(
        tmp_path,
        [
            {"framework": "jax", "success": True, "skipped": False},
            {
                "framework": "jax",
                "success": False,
                "skipped": True,
                "error": "Dependency not installed: jax",
            },
        ],
    )

    output = analyze_framework_outputs(tmp_path, LOGGER)

    jax = _summary(output, "jax")
    assert jax["status"] == "success_with_skips"
    assert jax["status_reason"] == "Dependency not installed: jax"


def test_legacy_detail_without_status_keys_degrades_as_failed(
    tmp_path: Path,
) -> None:
    _write_execution_summary(tmp_path, [{"framework": "jax"}])

    output = analyze_framework_outputs(tmp_path, LOGGER)

    jax = _summary(output, "jax")
    assert jax["status"] == "failed"
    assert jax["failed_count"] == 1
    assert output["comparisons"]["success_rates"] == {"jax": 0.0}


# ── Render-failure degradation ─────────────────────────────────────────────


def test_render_failure_only_framework_degrades_to_render_failed_column(
    tmp_path: Path,
) -> None:
    _write_execution_summary(
        tmp_path,
        [{"framework": "pymdp", "success": True, "skipped": False}],
        render_failures=[
            {
                "file": "input/model.md",
                "framework": "rxinfer",
                "message": "Artifact content no longer matches render receipt",
            }
        ],
    )

    output = analyze_framework_outputs(tmp_path, LOGGER)

    rxinfer = _summary(output, "rxinfer")
    assert rxinfer["status"] == "render_failed"
    assert rxinfer["status_reason"] == (
        "Artifact content no longer matches render receipt"
    )
    assert rxinfer["total_count"] == 0
    assert rxinfer["render_failures"][0]["file"] == "input/model.md"
    # The healthy framework's column is untouched.
    assert _summary(output, "pymdp")["status"] == "success"


def test_render_failures_attach_without_overriding_execution_status(
    tmp_path: Path,
) -> None:
    _write_execution_summary(
        tmp_path,
        [{"framework": "jax", "success": True, "skipped": False}],
        render_failures=[
            {
                "file": "input/other.md",
                "framework": "jax",
                "message": "render failed",
            }
        ],
    )

    output = analyze_framework_outputs(tmp_path, LOGGER)

    jax = _summary(output, "jax")
    assert jax["status"] == "success"
    assert jax["render_failures"][0]["message"] == "render failed"


def test_receipt_mismatch_all_framework_does_not_create_column(
    tmp_path: Path,
) -> None:
    _write_execution_summary(
        tmp_path,
        [{"framework": "pymdp", "success": True, "skipped": False}],
        render_failures=[
            {
                "file": "render_processing_summary.json",
                "framework": "all",
                "message": "Render receipt belongs to a different run",
            }
        ],
    )

    output = analyze_framework_outputs(tmp_path, LOGGER)

    assert set(output["frameworks"]) == {"pymdp"}


# ── Report rendering over degraded data ────────────────────────────────────


def test_comparison_report_renders_degraded_statuses(tmp_path: Path) -> None:
    _write_execution_summary(
        tmp_path,
        [
            {"framework": "pymdp", "success": True, "skipped": False},
            {
                "framework": "jax",
                "success": False,
                "skipped": True,
                "error": "Dependency not installed: jax",
            },
            {
                "framework": "discopy",
                "success": False,
                "skipped": False,
                "error": "Script failed with return code 2",
            },
        ],
        render_failures=[
            {
                "file": "input/model.md",
                "framework": "rxinfer",
                "message": "render receipt invalid",
            }
        ],
    )
    output = analyze_framework_outputs(tmp_path, LOGGER)

    report_file = generate_framework_comparison_report(output, tmp_path, LOGGER)

    report = Path(report_file).read_text()
    assert "- Status: success" in report
    assert "- Status: skipped" in report
    assert "- Status: failed" in report
    assert "- Status: render_failed" in report
    assert "- Status Reason: Dependency not installed: jax" in report
    assert "- Skipped Executions: 1" in report
    assert "- Failed Executions: 1" in report
    # The JSON twin carries the same statuses.
    json_data = json.loads(
        (tmp_path / "cross_framework" / "framework_comparison_data.json").read_text()
    )
    assert json_data["frameworks"]["rxinfer"]["status"] == "render_failed"


def test_missing_execution_summary_returns_empty_comparison(tmp_path: Path) -> None:
    output = analyze_framework_outputs(tmp_path, LOGGER)

    assert output["frameworks"] == {}
    assert output["comparisons"] == {}
    assert output["metrics"] == {}
