"""Real behavioral tests for the preflight diagnostics public API.

These exercise actual YAML parsing, report aggregation, and markdown
serialization without mocking the module under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.pipeline import preflight


def test_validate_config_missing_file_reports_warning(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    report = preflight.validate_config(missing)
    assert report.checks_failed == 0
    assert any(
        issue.severity == "warning" and "Config file not found" in issue.message
        for issue in report.issues
    )


def test_validate_config_valid_minimal_yaml_passes(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  model: smollm2\n  timeout_seconds: 60\n")
    report = preflight.validate_config(cfg)
    assert report.checks_failed == 0
    # file exists, valid YAML, llm.model, and llm.timeout_seconds all pass
    assert report.checks_passed >= 3
    assert report.is_ok


def test_validate_config_rejects_invalid_timeout(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  timeout_seconds: -5\n")
    report = preflight.validate_config(cfg)
    assert any(
        issue.severity == "error" and "Invalid llm.timeout_seconds" in issue.message
        for issue in report.issues
    )
    assert not report.is_ok


def test_validate_config_warns_on_huge_timeout(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  timeout_seconds: 99999\n")
    report = preflight.validate_config(cfg)
    assert any(
        issue.severity == "warning" and "Very large LLM timeout" in issue.message
        for issue in report.issues
    )


def test_run_preflight_combines_config_and_environment(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  model: m\n")
    report = preflight.run_preflight(cfg)
    # Config report produced passes; environment report merged in.
    assert report.checks_passed >= 1
    assert report.issues
    # Combined report must not raise.
    assert isinstance(report.to_markdown(), str)


def test_preflight_report_to_markdown_reports_success() -> None:
    report = preflight.PreflightReport()
    report.add_pass("Python ok")
    report.add_pass("Package numpy")
    md = report.to_markdown()
    assert "# Preflight Check Report" in md
    assert "🟢" in md
    assert "2 passed" in md


def test_preflight_report_to_markdown_lists_issues() -> None:
    report = preflight.PreflightReport()
    report.add_issue(
        "dependency", "error", "Package not found: foo", fix="pip install foo"
    )
    md = report.to_markdown()
    assert "🔴" in md
    assert "1 failed" in md
    assert "**[dependency]**" in md
    assert "Fix:" in md


def test_preflight_issue_fix_is_optional() -> None:
    report = preflight.PreflightReport()
    report.add_issue("config", "warning", "No fix provided")
    assert report.checks_failed == 0
    assert report.is_ok


# --- W2-D5: in-memory config validation (wired into main.py startup) --------


def test_validate_config_dict_flags_bad_skip_steps() -> None:
    """An in-memory bad skip_steps value must fail the report like the file path."""
    report = preflight.validate_config_dict({"pipeline": {"skip_steps": ["abc"]}})
    assert not report.is_ok
    assert any(
        issue.severity == "error" and "Invalid pipeline.skip_steps" in issue.message
        for issue in report.issues
    )


def test_validate_config_dict_passes_valid_skip_steps() -> None:
    report = preflight.validate_config_dict({"pipeline": {"skip_steps": [13]}})
    assert report.is_ok


def test_validate_config_dict_rejects_non_mapping() -> None:
    report = preflight.validate_config_dict(["not", "a", "mapping"])  # type: ignore[arg-type]
    assert not report.is_ok
    assert any("must be a YAML mapping" in issue.message for issue in report.issues)


def test_validate_config_file_aggregates_dict_findings(tmp_path: Path) -> None:
    """The file-based validator reuses the in-memory validator for section checks."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("pipeline:\n  skip_steps: [13]\n")
    report = preflight.validate_config(cfg)
    assert report.is_ok
    # "Config file exists" pass + "pipeline.skip_steps" pass.
    assert report.checks_passed >= 2


def test_validate_config_file_propagates_skip_step_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("pipeline:\n  skip_steps: [999]\n")
    report = preflight.validate_config(cfg)
    assert not report.is_ok
    assert any(
        issue.severity == "error" and "out of range" in issue.message
        for issue in report.issues
    )
