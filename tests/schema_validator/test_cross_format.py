#!/usr/bin/env python3
"""
Unit tests for gnn.schema_validator.cross_format (MED-T4).

Covers the observable public behavior of CrossFormatValidator,
CrossFormatValidationResult, and the two convenience functions:
consistency-rate accounting, format-issue grouping, end-to-end
cross-format validation of valid/invalid GNN content, and schema
definition consistency checks against a hermetic schemas directory.
"""

from pathlib import Path
from typing import Any

import pytest

from gnn.schema_validator.cross_format import (
    CrossFormatValidationResult,
    CrossFormatValidator,
    validate_cross_format_consistency,
    validate_schema_consistency,
)

EXEMPLAR = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/discrete/time_varying_dynamics.md"
)


@pytest.fixture
def exemplar_content() -> str:
    """The canonical valid GNN exemplar, read once per test."""
    return EXEMPLAR.read_text(encoding="utf-8")


@pytest.fixture
def validator() -> CrossFormatValidator:
    """Validator anchored at an empty module path (no schema files)."""
    return CrossFormatValidator(gnn_module_path=Path("/nonexistent-schemas"))


class TestCrossFormatValidationResult:
    """Result-object accounting: consistency rate and per-format issues."""

    def test_consistency_rate_empty_is_zero(self) -> Any:
        result = CrossFormatValidationResult(is_consistent=True)
        assert result.get_consistency_rate() == 0.0

    def test_consistency_rate_counts_valid_formats(self) -> Any:
        from gnn.types import ValidationResult

        result = CrossFormatValidationResult(
            is_consistent=True,
            schema_formats=["json", "yaml", "xml"],
            format_results={
                "json": ValidationResult(is_valid=True),
                "yaml": ValidationResult(is_valid=True),
                "xml": ValidationResult(is_valid=False, errors=["bad"]),
            },
        )
        # xml is in schema_formats but has no passing format_result entry.
        result.format_results["xml_missing"] = ValidationResult(is_valid=True)
        assert result.get_consistency_rate() == pytest.approx(2 / 3 * 100)

    def test_add_format_issue_groups_per_format(self) -> Any:
        result = CrossFormatValidationResult(is_consistent=True)
        result.add_format_issue("json", "e1")
        result.add_format_issue("json", "e2")
        result.add_format_issue("yaml", "e3")
        assert result.format_specific_issues == {
            "json": ["e1", "e2"],
            "yaml": ["e3"],
        }


class TestCrossFormatValidation:
    """End-to-end validate_cross_format_consistency on real GNN content."""

    def test_valid_exemplar_is_consistent(
        self, validator: CrossFormatValidator, exemplar_content: str
    ) -> Any:
        result = validator.validate_cross_format_consistency(exemplar_content)
        assert result.is_consistent, result.inconsistencies
        assert set(result.schema_formats) == {"binary", "markdown"}
        assert {fmt: r.is_valid for fmt, r in result.format_results.items()} == {
            "binary": True,
            "markdown": True,
        }
        assert result.get_consistency_rate() == 100.0
        assert result.metadata["source_format"] == "markdown"
        assert result.metadata["formats_tested"] == 2
        assert result.metadata["valid_formats"] == 2
        assert result.metadata["invalid_formats"] == 0
        assert result.performance_metrics["total_validation_time"] >= 0.0

    def test_non_gnn_content_is_inconsistent(
        self, validator: CrossFormatValidator
    ) -> Any:
        result = validator.validate_cross_format_consistency("just some prose\n")
        assert result.is_consistent is False
        assert result.format_results["markdown"].is_valid is False
        assert result.format_results["binary"].is_valid is True
        assert result.inconsistencies, "missing-section errors must surface"
        assert "markdown" in result.format_specific_issues
        assert result.metadata["invalid_formats"] == 1

    def test_empty_content_is_inconsistent(
        self, validator: CrossFormatValidator
    ) -> Any:
        result = validator.validate_cross_format_consistency("")
        assert result.is_consistent is False
        assert result.format_results["markdown"].is_valid is False

    def test_validate_over_file_list(
        self, validator: CrossFormatValidator, tmp_path: Path
    ) -> Any:
        bad_file = tmp_path / "bad.md"
        bad_file.write_text("not a gnn document\n", encoding="utf-8")
        report = validator.validate([EXEMPLAR, bad_file])
        assert report["files_validated"] == 2
        assert report["success"] is False
        assert set(report["results"]) == {str(EXEMPLAR), str(bad_file)}
        assert report["results"][str(EXEMPLAR)].is_consistent is True
        assert report["results"][str(bad_file)].is_consistent is False

    def test_convenience_function_returns_validator_result(
        self, exemplar_content: str
    ) -> Any:
        result = validate_cross_format_consistency(exemplar_content)
        assert isinstance(result, CrossFormatValidationResult)
        assert result.is_consistent, result.inconsistencies
        assert result.format_results["markdown"].is_valid is True


class TestSchemaDefinitionsConsistency:
    """Schema-artifact consistency checks against a hermetic schemas dir."""

    GOOD_SECTIONS = (
        "GNNSection",
        "GNNVersionAndFlags",
        "ModelName",
        "ModelAnnotation",
        "StateSpaceBlock",
        "Connections",
        "InitialParameterization",
        "Time",
        "Footer",
    )

    def _write_schemas(self, tmp_path: Path, **schemas: str) -> Path:
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        for name, text in schemas.items():
            (schemas_dir / name).write_text(text, encoding="utf-8")
        return tmp_path

    def test_missing_schemas_warn_but_stay_consistent(self, tmp_path: Path) -> Any:
        validator = CrossFormatValidator(gnn_module_path=tmp_path)
        result = validator.validate_schema_definitions_consistency()
        assert result.is_consistent is True
        assert result.schema_formats == []
        assert any("Schema file not found" in w for w in result.warnings)

    def test_aligned_json_and_yaml_schemas_are_consistent(self, tmp_path: Path) -> Any:
        import json

        json_schema = {
            "properties": {s: {"type": "string"} for s in self.GOOD_SECTIONS}
        }
        yaml_schema = "required_sections:\n" + "".join(
            f"  - {s}\n" for s in self.GOOD_SECTIONS
        )
        gnn_path = self._write_schemas(
            tmp_path,
            **{
                "json.json": json.dumps(json_schema),
                "yaml.yaml": yaml_schema,
            },
        )
        validator = CrossFormatValidator(gnn_module_path=gnn_path)
        result = validator.validate_schema_definitions_consistency()
        assert result.is_consistent is True, result.inconsistencies
        assert result.schema_formats == ["json", "yaml"]
        assert any(
            "Good structural consistency between JSON and YAML schemas" in w
            for w in result.warnings
        )
        assert any("json schema has good coverage" in w for w in result.warnings)

    def test_divergent_json_and_yaml_schemas_are_inconsistent(
        self, tmp_path: Path
    ) -> Any:
        import json

        json_schema = {
            "properties": {s: {"type": "string"} for s in self.GOOD_SECTIONS}
        }
        yaml_schema = "required_sections:\n  - ModelName\n"
        gnn_path = self._write_schemas(
            tmp_path,
            **{
                "json.json": json.dumps(json_schema),
                "yaml.yaml": yaml_schema,
            },
        )
        validator = CrossFormatValidator(gnn_module_path=gnn_path)
        result = validator.validate_schema_definitions_consistency()
        assert result.is_consistent is False
        assert any(
            "Significant structural differences between JSON and YAML schemas" in i
            for i in result.inconsistencies
        )

    def test_unloadable_json_schema_is_inconsistent(self, tmp_path: Path) -> Any:
        gnn_path = self._write_schemas(tmp_path, **{"json.json": "{not valid json"})
        validator = CrossFormatValidator(gnn_module_path=gnn_path)
        result = validator.validate_schema_definitions_consistency()
        assert result.is_consistent is False
        assert any("Failed to load json schema" in i for i in result.inconsistencies)

    def test_convenience_function_checks_repo_schemas(self) -> Any:
        result = validate_schema_consistency()
        assert isinstance(result, CrossFormatValidationResult)
        # The repo currently ships no schemas/ directory next to the module,
        # so every lookup is a warning and consistency is trivially preserved.
        assert result.is_consistent is True
        assert result.schema_formats == []
