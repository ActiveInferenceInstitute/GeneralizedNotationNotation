import pytest
import os
from type_checker.checker import GNNTypeChecker

class TestTypeCheckerOverall:
    """Test suite for Type Checker module."""

    @pytest.fixture
    def valid_gnn_file(self, safe_filesystem):
        content = """
# Valid GNN Example: MyModel

## GNNSection
Reflects=ActiveInference

## GNNVersionAndFlags
GNN v1.0
Flags: strict=false

## ModelName
MyModel

## StateSpaceBlock
s[1, type=float]

## Connections
s>s

## Time
Static
"""
        return safe_filesystem.create_file("valid_model.md", content)

    @pytest.fixture
    def type_error_gnn_file(self, safe_filesystem):
        content = """
# Invalid GNN Example: ErrorModel

## GNNVersionAndFlags
GNN v1.0

## ModelName
ErrorModel

## StateSpaceBlock
s[1, type=float]

## Connections
x->s  # x is undefined
"""
        return safe_filesystem.create_file("error_model.md", content)

    def test_check_file_valid(self, valid_gnn_file):
        """Test checking a valid file."""
        checker = GNNTypeChecker(strict_mode=False)
        is_valid, errors, warnings, details = checker.check_file(str(valid_gnn_file))
        
        assert is_valid is True
        assert len(errors) == 0

    def test_check_file_strict_mode(self, valid_gnn_file):
        """Test strict mode."""
        checker = GNNTypeChecker(strict_mode=True)
        is_valid, errors, warnings, details = checker.check_file(str(valid_gnn_file))
        
        # Depending on strict rules, it might still pass or fail if something small is missing
        # Based on my read of checking logic, 'valid_model.md' should be mostly fine.
        assert is_valid is True

    def test_check_file_with_errors(self, type_error_gnn_file):
        """Test detecting undefined variables."""
        checker = GNNTypeChecker(strict_mode=True)
        is_valid, errors, warnings, details = checker.check_file(str(type_error_gnn_file))
        
        # Check source code logic:
        # _check_connections adds WARNING: "Connection references potentially undefined variable: x"
        # In check_file: if strict_mode and critical_warnings > 0, it promotes to ERROR and fails.
        # "Connection references potentially undefined variable" IS in critical_warning_patterns.
        
        assert is_valid is False
        assert any("undefined variable" in e for e in errors)

    def test_check_directory(self, safe_filesystem, valid_gnn_file):
        """Test directory scanning."""
        checker = GNNTypeChecker()
        results = checker.check_directory(str(safe_filesystem.temp_dir))
        
        assert str(valid_gnn_file) in results
        assert results[str(valid_gnn_file)]["is_valid"] is True
