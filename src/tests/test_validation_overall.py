import pytest
from validation.semantic_validator import SemanticValidator, process_semantic_validation

class TestValidationOverall:
    """Test suite for Validation module."""

    @pytest.fixture
    def valid_gnn_content(self):
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
    def invalid_gnn_content(self):
        return """
# Invalid Model missing the required block keyword
ModelName: Invalid
"""

    def test_semantic_validator_valid_model(self, valid_gnn_content):
        """Test validation of a syntactically correct model."""
        validator = SemanticValidator(validation_level="standard")
        result = validator.validate(valid_gnn_content)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_semantic_validator_invalid_model(self, invalid_gnn_content):
        """Test validation of a broken model."""
        validator = SemanticValidator(validation_level="standard")
        result = validator.validate(invalid_gnn_content)
        
        assert result["is_valid"] is False
        assert any("Missing StateSpaceBlock" in e for e in result["errors"])

    def test_process_semantic_validation_wrapper(self, safe_filesystem, valid_gnn_content):
        """Test the top-level processing wrapper."""
        file_path = safe_filesystem.create_file("model.md", valid_gnn_content)
        
        result = process_semantic_validation(file_path)
        
        assert result["file_name"] == "model.md"
        assert result["valid"] is True
        assert result["semantic_score"] > 0.9  # Should be high for valid model

    def test_validation_levels(self, valid_gnn_content):
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
