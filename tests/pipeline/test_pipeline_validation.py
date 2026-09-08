"""Unit tests for ``gnn.pipeline.pipeline_validation``.

The module previously had zero test coverage. These tests exercise the
pure-function core (module-import validators, output-structure check,
recommendation generation, and the ``generate_validation_report``
orchestration) against synthetic module trees in ``tmp_path``. The
repo-scanning validators invoked inside ``generate_validation_report``
(configuration/argument/dependency scans) read the real fixed tree, so
summary assertions are self-consistency pins (status derived from the
report's own counters) instead of assumptions about their results.

Deliberately unmarked: offline unit logic for the default suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gnn.pipeline.pipeline_validation import (
    EXPECTED_OUTPUTS,
    generate_improvement_recommendations,
    generate_validation_report,
    get_pipeline_modules,
    validate_centralized_imports,
    validate_module_imports,
    validate_output_structure,
)


def _write_module(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.unit
class TestGetPipelineModules:
    def test_discovers_numbered_steps_and_main_sorted(self, tmp_path: Path) -> None:
        _write_module(tmp_path / "7_export.py", "# step 7\n")
        _write_module(tmp_path / "3_gnn.py", "# step 3\n")
        _write_module(tmp_path / "main.py", "# orchestrator\n")
        _write_module(tmp_path / "helper.py", "# not a numbered step\n")

        modules = get_pipeline_modules(tmp_path)

        names = [m.name for m in modules]
        assert names == ["3_gnn.py", "7_export.py", "main.py"]

    def test_empty_src_dir_returns_empty_list(self, tmp_path: Path) -> None:
        assert get_pipeline_modules(tmp_path) == []


@pytest.mark.unit
class TestValidateModuleImports:
    def test_flags_missing_utils_import_and_step_logging(self, tmp_path: Path) -> None:
        module = _write_module(tmp_path / "3_gnn.py", "print('step')\n")

        issues = validate_module_imports(module)

        assert "Missing centralized utils import" in issues["errors"]
        assert "Missing setup_step_logging call" in issues["errors"]

    def test_main_py_requires_setup_main_logging(self, tmp_path: Path) -> None:
        module = _write_module(
            tmp_path / "main.py", "from utils import ArgumentParser\n"
        )

        issues = validate_module_imports(module)

        assert issues["errors"] == ["Missing setup_main_logging call"]

    def test_suggestion_for_string_first_log_call(self, tmp_path: Path) -> None:
        module = _write_module(
            tmp_path / "3_gnn.py",
            'from utils import setup_step_logging\nlog_step_start("step 3")\n',
        )

        issues = validate_module_imports(module)

        assert any("log_step_start" in s for s in issues["suggestions"])

    def test_unreadable_module_recorded_as_error(self, tmp_path: Path) -> None:
        issues = validate_module_imports(tmp_path / "does-not-exist.py")

        assert len(issues["errors"]) == 1
        assert issues["errors"][0].startswith("Failed to read module:")


@pytest.mark.unit
class TestValidateCentralizedImports:
    def test_warns_about_missing_recommended_imports(self, tmp_path: Path) -> None:
        module = _write_module(
            tmp_path / "3_gnn.py", "from utils import setup_step_logging\n"
        )

        issues = validate_centralized_imports(module)

        assert issues["errors"] == []
        warning = issues["warnings"][0]
        assert warning.startswith("Missing recommended imports:")
        assert "log_step_start" in warning and "log_step_error" in warning

    def test_performance_tracking_suggestion_for_compute_steps(
        self, tmp_path: Path
    ) -> None:
        module = _write_module(
            tmp_path / "11_render.py",
            "from utils import setup_step_logging\nfrom pipeline import x\n",
        )

        issues = validate_centralized_imports(module)

        assert any("performance tracking" in s for s in issues["suggestions"])

    def test_hardcoded_absolute_path_recorded_as_improvement(
        self, tmp_path: Path
    ) -> None:
        module = _write_module(
            tmp_path / "3_gnn.py",
            "from utils import setup_step_logging\nout = Path('/absolute/path')\n",
        )

        issues = validate_centralized_imports(module)

        assert "Consider using centralized path configuration" in issues["improvements"]


@pytest.mark.unit
class TestValidateOutputStructure:
    def test_missing_directory_reported(self, tmp_path: Path) -> None:
        issues = validate_output_structure(tmp_path / "does-not-exist")

        assert issues["missing"] == ["Output directory does not exist"]
        assert issues["present"] == []

    def test_all_expected_outputs_present(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        for files in EXPECTED_OUTPUTS.values():
            for expected in files:
                (output_dir / expected).parent.mkdir(parents=True, exist_ok=True)
                (output_dir / expected).touch()

        issues = validate_output_structure(output_dir)

        assert issues["missing"] == []
        assert len(issues["present"]) == sum(
            len(files) for files in EXPECTED_OUTPUTS.values()
        )


@pytest.mark.unit
class TestGenerateImprovementRecommendations:
    def test_recommends_fixing_error_and_warning_modules(self) -> None:
        report = {
            "module_issues": {
                "a.py": {"errors": ["missing utils"], "warnings": []},
                "b.py": {"errors": [], "warnings": ["missing import"]},
            }
        }

        recommendations = generate_improvement_recommendations(report)

        assert any("Fix import errors in 1 modules" in r for r in recommendations)
        assert any("Address warnings in 1 modules" in r for r in recommendations)

    def test_empty_report_yields_no_module_recommendations(self) -> None:
        recommendations = generate_improvement_recommendations({"module_issues": {}})

        assert not any("Fix import errors" in r for r in recommendations)


@pytest.mark.unit
class TestGenerateValidationReport:
    def test_report_shape_and_module_accounting(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write_module(
            src_dir / "3_gnn.py",
            "from utils import (setup_step_logging, log_step_start, log_step_success,"
            " log_step_warning, log_step_error)\n"
            "# uses pipeline configuration\n",
        )
        output_dir = tmp_path / "output"

        report: dict[str, Any] = generate_validation_report(src_dir, output_dir)

        assert set(report) >= {
            "modules_checked",
            "modules_with_issues",
            "output_validation",
            "module_issues",
            "configuration_validation",
            "argument_validation",
            "dependency_validation",
            "performance_tracking_coverage",
            "improvement_recommendations",
            "summary",
        }
        assert report["modules_checked"] == 1
        assert report["module_issues"] == {}
        assert isinstance(report["output_validation"]["naming_violations"], list)
        summary = report["summary"]
        assert summary["total_modules"] == 1
        assert summary["modules_with_issues"] == 0
        assert summary["missing_outputs"] == len(report["output_validation"]["missing"])
        # Status derives from the report's own counters (the repo-wide config
        # scan result is environmental).
        expected_status = "FAIL" if summary["configuration_errors"] > 0 else "WARN"
        assert summary["status"] == expected_status

    def test_module_with_import_errors_is_flagged_and_fails(
        self, tmp_path: Path
    ) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write_module(src_dir / "3_gnn.py", "print('missing everything')\n")
        output_dir = tmp_path / "output"

        report = generate_validation_report(src_dir, output_dir)

        assert report["modules_with_issues"] == 1
        assert "3_gnn.py" in report["module_issues"]
        assert report["summary"]["status"] == "FAIL"
        assert any(
            "Fix import errors in 1 modules" in r
            for r in report["improvement_recommendations"]
        )
