import os
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from type_checker.checker import GNNTypeChecker

class TestTypeCheckerOverall:
    """Test suite for Type Checker module."""

    @pytest.fixture
    def valid_gnn_file(self, safe_filesystem: Any) -> Path:
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
    def type_error_gnn_file(self, safe_filesystem: Any) -> Path:
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

    def test_check_file_valid(self, valid_gnn_file: Path) -> None:
        """Test checking a valid file."""
        checker = GNNTypeChecker(strict_mode=False)
        is_valid, errors, warnings, details = checker.check_file(str(valid_gnn_file))

        assert is_valid is True
        assert len(errors) == 0

    def test_check_file_strict_mode(self, valid_gnn_file: Path) -> None:
        """Test strict mode."""
        checker = GNNTypeChecker(strict_mode=True)
        is_valid, errors, warnings, details = checker.check_file(str(valid_gnn_file))

        # Depending on strict rules, it might still pass or fail if something small is missing
        # Based on my read of checking logic, 'valid_model.md' should be mostly fine.
        assert is_valid is True

    def test_check_file_with_errors(self, type_error_gnn_file: Path) -> None:
        """Test detecting undefined variables."""
        checker = GNNTypeChecker(strict_mode=True)
        is_valid, errors, warnings, details = checker.check_file(str(type_error_gnn_file))

        # Check source code logic:
        # _check_connections adds WARNING: "Connection references potentially undefined variable: x"
        # In check_file: if strict_mode and critical_warnings > 0, it promotes to ERROR and fails.
        # "Connection references potentially undefined variable" IS in critical_warning_patterns.

        assert is_valid is False
        assert any("undefined variable" in e for e in errors)

    def test_check_directory(self, safe_filesystem: Any, valid_gnn_file: Path) -> None:
        """Test directory scanning."""
        checker = GNNTypeChecker()
        results = checker.check_directory(str(safe_filesystem.temp_dir))

        assert str(valid_gnn_file) in results
        assert results[str(valid_gnn_file)]["is_valid"] is True

    def test_check_nonexistent_file_returns_error(self):
        """Checking a nonexistent file should return is_valid=False with an error message."""
        checker = GNNTypeChecker()
        is_valid, errors, warnings, details = checker.check_file("/nonexistent/path/model.md")

        assert is_valid is False
        assert len(errors) > 0

    def test_check_corrupted_file_returns_error(self, safe_filesystem):
        """Checking a file with binary/corrupted content should return is_valid=False."""
        corrupted = safe_filesystem.create_file("corrupted.md", "\x00\x01\x02\x03binary garbage")
        checker = GNNTypeChecker()
        is_valid, errors, warnings, details = checker.check_file(str(corrupted))

        # A file with no recognisable GNN sections is invalid (missing required sections).
        assert is_valid is False

    def test_check_file_missing_required_sections(self, safe_filesystem):
        """A file missing required GNN sections should be invalid."""
        content = "# Just a title\n\nSome random text with no GNN sections.\n"
        incomplete = safe_filesystem.create_file("incomplete.md", content)
        checker = GNNTypeChecker(strict_mode=False)
        is_valid, errors, warnings, details = checker.check_file(str(incomplete))

        assert is_valid is False

    def test_warnings_not_errors_in_non_strict_mode(self, safe_filesystem):
        """Undefined-variable connection produces a warning in non-strict mode, not an error."""
        content = """
## GNNSection
Reflects=ActiveInference

## ModelName
WarnModel

## StateSpaceBlock
s[1, type=float]

## Connections
x->s
"""
        warn_file = safe_filesystem.create_file("warn_model.md", content)
        checker = GNNTypeChecker(strict_mode=False)
        is_valid, errors, warnings, details = checker.check_file(str(warn_file))

        # In non-strict mode undefined-variable warnings do NOT escalate to errors.
        assert any("undefined variable" in w for w in warnings)
        assert not any("undefined variable" in e for e in errors)

    @pytest.mark.skipif(os.getuid() == 0, reason="root can read any file")
    def test_check_unreadable_file_returns_error(self, safe_filesystem):
        """Checking a file without read permission should return is_valid=False."""
        locked = safe_filesystem.create_file("locked.md", "## ModelName\nLocked\n")
        locked.chmod(0o000)
        try:
            checker = GNNTypeChecker()
            is_valid, errors, warnings, details = checker.check_file(str(locked))
            assert is_valid is False
            assert len(errors) > 0
        finally:
            locked.chmod(0o644)
