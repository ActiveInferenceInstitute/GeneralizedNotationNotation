"""Tests for CrossFormatValidator and related classes in gnn/cross_format_validator.py."""
import pytest

from gnn.cross_format_validator import CrossFormatValidationResult, CrossFormatValidator
from gnn.types import ValidationResult

# ── CrossFormatValidationResult ────────────────────────────────────────────

class TestCrossFormatValidationResult:
    def test_default_consistency_rate_no_formats(self) -> None:
        """get_consistency_rate() returns 0.0 when schema_formats is empty."""
        r = CrossFormatValidationResult(is_consistent=True)
        assert r.get_consistency_rate() == 0.0

    def test_consistency_rate_all_valid(self) -> None:
        """100% when every format in schema_formats has a valid ValidationResult."""
        vr = ValidationResult(is_valid=True)
        r = CrossFormatValidationResult(
            is_consistent=True,
            schema_formats=["json", "yaml"],
            format_results={"json": vr, "yaml": vr},
        )
        assert r.get_consistency_rate() == pytest.approx(100.0)

    def test_consistency_rate_partial(self) -> None:
        """50% when half the formats are valid."""
        valid = ValidationResult(is_valid=True)
        invalid = ValidationResult(is_valid=False)
        r = CrossFormatValidationResult(
            is_consistent=False,
            schema_formats=["json", "yaml"],
            format_results={"json": valid, "yaml": invalid},
        )
        assert r.get_consistency_rate() == pytest.approx(50.0)

    def test_consistency_rate_format_not_in_results(self) -> None:
        """Format listed in schema_formats but absent from format_results counts as 0."""
        r = CrossFormatValidationResult(
            is_consistent=False,
            schema_formats=["json"],
            format_results={},  # json result missing
        )
        assert r.get_consistency_rate() == 0.0

    def test_add_format_issue_creates_list(self) -> None:
        r = CrossFormatValidationResult(is_consistent=True)
        r.add_format_issue("json", "missing field x")
        assert r.format_specific_issues["json"] == ["missing field x"]

    def test_add_format_issue_appends(self) -> None:
        r = CrossFormatValidationResult(is_consistent=True)
        r.add_format_issue("json", "issue 1")
        r.add_format_issue("json", "issue 2")
        assert len(r.format_specific_issues["json"]) == 2


# ── CrossFormatValidator ───────────────────────────────────────────────────

class TestCrossFormatValidator:
    def test_instantiation_default(self) -> None:
        v = CrossFormatValidator()
        assert v is not None
        assert v.enable_round_trip_testing is False

    def test_instantiation_with_round_trip(self) -> None:
        v = CrossFormatValidator(enable_round_trip_testing=True)
        assert v.enable_round_trip_testing is True

    def test_validate_cross_format_consistency_returns_result(self) -> None:
        """validate_cross_format_consistency always returns a CrossFormatValidationResult."""
        v = CrossFormatValidator()
        minimal_gnn = "## GNNSection\nActInfPOMDP\n\n## ModelName\nTest\n"
        result = v.validate_cross_format_consistency(minimal_gnn)
        assert isinstance(result, CrossFormatValidationResult)

    def test_validate_cross_format_consistency_has_metadata(self) -> None:
        """Result always has source_format metadata."""
        v = CrossFormatValidator()
        result = v.validate_cross_format_consistency("", source_format="markdown")
        assert result.metadata.get("source_format") == "markdown"

    def test_validate_cross_format_schema_files_absent_no_crash(self) -> None:
        """When schema files are absent, validation returns a result with warnings but does not crash."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            # Point validator at an empty directory — no schema files present
            v = CrossFormatValidator(gnn_module_path=Path(tmpdir))
            result = v.validate_cross_format_consistency("## GNNSection\nActInfPOMDP\n")
        # Must not raise; result must be a proper object
        assert isinstance(result, CrossFormatValidationResult)

    def test_validate_cross_format_consistency_is_bool(self) -> None:
        """is_consistent field is always a bool."""
        v = CrossFormatValidator()
        result = v.validate_cross_format_consistency("")
        assert isinstance(result.is_consistent, bool)
