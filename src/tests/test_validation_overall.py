import pytest
from typing import Any, Dict, List, Optional
from validation.semantic_validator import SemanticValidator, process_semantic_validation

class TestValidationOverall:
    """Test suite for Validation module."""

    @pytest.fixture
    def valid_gnn_content(self) -> str:
        return """
# Valid Model
ModelName: TestModel

StateSpaceBlock {
    Name: block1
    Dimensions: 1
}

StateSpaceBlock {
    Name: block2
    Dimensions: 1
}

Connection {
    From: block1
    To: block2
}
"""

    @pytest.fixture
    def invalid_gnn_content(self) -> str:
        return """
# Invalid Model missing the required block keyword
ModelName: Invalid
"""

    def test_semantic_validator_valid_model(self, valid_gnn_content: str) -> None:
        """Test validation of a syntactically correct model."""
        validator = SemanticValidator(validation_level="standard")
        result = validator.validate(valid_gnn_content)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_semantic_validator_invalid_model(self, invalid_gnn_content: str) -> None:
        """Test validation of a broken model."""
        validator = SemanticValidator(validation_level="standard")
        result = validator.validate(invalid_gnn_content)

        assert result["is_valid"] is False
        assert any("Missing StateSpaceBlock" in e for e in result["errors"])

    def test_process_semantic_validation_wrapper(self, safe_filesystem: Any, valid_gnn_content: str) -> None:
        """Test the top-level processing wrapper."""
        file_path = safe_filesystem.create_file("model.md", valid_gnn_content)

        result = process_semantic_validation(file_path)

        assert result["file_name"] == "model.md"
        assert result["valid"] is True
        assert result["semantic_score"] > 0.9  # Should be high for valid model

    def test_validation_levels(self, valid_gnn_content: str) -> None:
        """Ensure strict mode catches subtle issues (missing active inference components)."""
        # The valid content above is missing 'Observation', 'Transition', 'Prior' keywords required by "strict"
        # validation for Active Inference principles.

        validator_strict = SemanticValidator(validation_level="strict")
        result = validator_strict.validate(valid_gnn_content)

        # It might still be "valid" in terms of structure, but should have warnings
        # The code shows warnings append to warnings list, errors prevent validity?
        # Let's check the code: keys are "errors" and "warnings". "is_valid" depends on errors count == 0.
        # But principles checker appends to WARNINGS not errors in `_validate_active_inference_principles`.

        # So it should be valid but have warnings.
        assert result["is_valid"] is True

        warning_texts = " ".join(result["warnings"])
        assert "Active Inference model missing explicit observation model" in warning_texts


# ── Validation sub-module smoke tests ────────────────────────────────────────

class TestConsistencyChecker:
    def test_module_importable(self):
        from validation import consistency_checker  # noqa: F401

    def test_class_instantiable(self):
        from validation.consistency_checker import ConsistencyChecker
        checker = ConsistencyChecker()
        assert checker is not None

    def test_check_consistency_with_dict(self):
        from validation.consistency_checker import check_consistency
        result = check_consistency({"ModelName": "TestModel", "StateSpaceBlock": "s[1]"})
        assert isinstance(result, dict)
        assert "consistent" in result or "is_consistent" in result

    def test_check_consistency_empty_dict(self):
        from validation.consistency_checker import check_consistency
        result = check_consistency({})
        assert isinstance(result, dict)


class TestValidationMCP:
    def test_module_importable(self):
        from validation import mcp  # noqa: F401

    def test_validate_gnn_file_mcp_nonexistent(self, tmp_path):
        from validation.mcp import validate_gnn_file_mcp
        result = validate_gnn_file_mcp(str(tmp_path / "nonexistent.md"))
        assert isinstance(result, dict)
        assert "success" in result or "error" in result

    def test_check_schema_compliance_mcp_empty(self):
        from validation.mcp import check_schema_compliance_mcp
        result = check_schema_compliance_mcp("")
        assert isinstance(result, dict)


class TestPerformanceProfiler:
    def test_module_importable(self):
        from validation import performance_profiler  # noqa: F401

    def test_class_instantiable(self):
        from validation.performance_profiler import PerformanceProfiler
        profiler = PerformanceProfiler()
        assert profiler is not None

    def test_profile_performance_with_string(self):
        from validation.performance_profiler import profile_performance
        result = profile_performance("## ModelName\nTestModel\n")
        assert isinstance(result, dict)
        assert "performance_score" in result or "error" in result

    def test_profile_performance_empty(self):
        from validation.performance_profiler import profile_performance
        result = profile_performance("")
        assert isinstance(result, dict)


class TestValidationInit:
    def test_module_importable(self):
        from validation import __init__  # noqa: F401

    def test_process_validation_empty_dir(self, tmp_path):
        from validation import process_validation
        result = process_validation(tmp_path, tmp_path / "out")
        assert isinstance(result, bool)
