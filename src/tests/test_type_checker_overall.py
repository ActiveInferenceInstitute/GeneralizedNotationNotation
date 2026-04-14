import os
from pathlib import Path
from typing import Any

import pytest

from type_checker.processor import GNNTypeChecker


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
s[3,1]

## StateSpaceBlock
s[5,1]  # Duplicate!
"""
        return safe_filesystem.create_file("error_model.md", content)

    def test_check_file_valid(self, valid_gnn_file: Path) -> None:
        """Test checking a valid file natively using processor layer."""
        checker = GNNTypeChecker()
        result = checker.validate_single_gnn_file(valid_gnn_file)

        assert result["valid"] is True
        assert len(result.get("errors", [])) == 0

    def test_check_file_with_errors(self, type_error_gnn_file: Path) -> None:
        """Test detecting duplicate/conflicting types."""
        checker = GNNTypeChecker()
        result = checker.validate_single_gnn_file(type_error_gnn_file)

        assert result["valid"] is False
        assert any("Duplicate" in e for e in result.get("errors", []))

    def test_check_directory(self, safe_filesystem: Any, valid_gnn_file: Path) -> None:
        """Test active processing architecture directory generation."""
        checker = GNNTypeChecker()
        out_dir = Path(safe_filesystem.temp_dir) / "out"
        out_dir.mkdir()
        
        # Processor layer evaluates directories inherently through validate_gnn_files
        success = checker.validate_gnn_files(Path(safe_filesystem.temp_dir), out_dir)

        assert success is True
        assert (out_dir / "type_check_results.json").exists()

    def test_check_nonexistent_file_returns_error(self):
        """Checking a nonexistent file gracefully handles failure."""
        checker = GNNTypeChecker()
        result = checker.validate_single_gnn_file(Path("/nonexistent/path/model.md"))

        assert result["valid"] is False
        assert len(result.get("errors", [])) > 0

    @pytest.mark.skipif(os.getuid() == 0, reason="root can read any file")
    def test_check_unreadable_file_returns_error(self, safe_filesystem):
        """Checking a file without read permission should return invalid safely."""
        locked = safe_filesystem.create_file("locked.md", "## ModelName\nLocked\n")
        locked.chmod(0o000)
        try:
            checker = GNNTypeChecker()
            result = checker.validate_single_gnn_file(locked)
            assert result["valid"] is False
            assert len(result.get("errors", [])) > 0
        finally:
            locked.chmod(0o644)
