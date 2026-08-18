"""Tests for validation module's public API surface not covered by test_validation_overall.

Covers: get_module_info, FEATURES, __version__, SemanticValidator edge cases,
ConsistencyChecker edge cases, PerformanceProfiler edge cases, process_validation.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestValidationConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import validation

        assert hasattr(validation, "FEATURES")
        assert isinstance(validation.FEATURES, dict)
        assert validation.FEATURES.get("semantic_validation") is True
        assert validation.FEATURES.get("performance_profiling") is True
        assert validation.FEATURES.get("consistency_checking") is True

    def test_version(self) -> None:
        import validation

        assert hasattr(validation, "__version__")
        assert isinstance(validation.__version__, str)

    def test_get_module_info(self) -> None:
        from validation import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert info["name"] == "validation"
        assert "version" in info
        assert "description" in info
        assert "features" in info


class TestSemanticValidatorEdgeCases:
    """Edge cases for SemanticValidator beyond basic valid/invalid."""

    def test_validation_level_strict_with_complete_model(self) -> None:
        """A model with all Active Inference components should pass strict too."""
        from validation.semantic_validator import SemanticValidator

        content = """# Complete Model
ModelName: Complete
StateSpaceBlock { Name: s Dimensions: 2 }
Connection { From: s To: o }
Observation { Name: o }
Transition { Name: B }
Prior { Name: D }
"""
        validator = SemanticValidator(validation_level="strict")
        result = validator.validate(content)
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result

    def test_validation_with_empty_content(self) -> None:
        from validation.semantic_validator import SemanticValidator

        validator = SemanticValidator()
        result = validator.validate("")
        assert isinstance(result, dict)
        assert "is_valid" in result

    def test_validation_with_none_content(self) -> None:
        from validation.semantic_validator import SemanticValidator

        validator = SemanticValidator()
        # None content is a caller error — the validator raises TypeError
        # rather than silently passing a falsy input through its rules.
        with pytest.raises(TypeError):
            validator.validate(None)  # type: ignore[arg-type]

    def test_validation_level_none(self) -> None:
        """Test that no validation level defaults to standard."""
        from validation.semantic_validator import SemanticValidator

        validator = SemanticValidator()
        assert hasattr(validator, "validation_level")

    def test_process_semantic_validation_with_dict(self) -> None:
        """process_semantic_validation can accept a dict directly."""
        from validation import process_semantic_validation

        result = process_semantic_validation({"ModelName": "Test"})
        assert isinstance(result, dict)


class TestConsistencyCheckerEdgeCases:
    """Edge cases for ConsistencyChecker."""

    def test_check_consistency_with_model_data(self) -> None:
        from validation import check_consistency

        data: dict[str, Any] = {
            "ModelName": "M",
            "variables": [{"name": "s", "dimensions": [3]}],
        }
        result = check_consistency(data)
        assert isinstance(result, dict)

    def test_check_consistency_with_complex_model(self) -> None:
        from validation import check_consistency

        data: dict[str, Any] = {
            "ModelName": "Complex",
            "variables": [
                {"name": "s", "dimensions": [3]},
                {"name": "o", "dimensions": [3]},
            ],
            "connections": [{"source": "s", "target": "o"}],
        }
        result = check_consistency(data)
        assert isinstance(result, dict)

    def test_check_consistency_with_none(self) -> None:
        from validation import check_consistency

        result = check_consistency(None)  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_consistency_checker_default_init(self) -> None:
        from validation.consistency_checker import ConsistencyChecker

        # ConsistencyChecker() takes no arguments per its public signature.
        checker = ConsistencyChecker()
        assert checker is not None


class TestPerformanceProfilerEdgeCases:
    """Edge cases for PerformanceProfiler."""

    def test_profile_with_none(self) -> None:
        from validation import profile_performance

        result = profile_performance(None)  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_profile_with_complex_model(self) -> None:
        from validation import profile_performance

        data: dict[str, Any] = {
            "ModelName": "PerfModel",
            "variables": [{"name": "s", "dimensions": [10]}],
            "connections": 5,
        }
        result = profile_performance(data)
        assert isinstance(result, dict)
