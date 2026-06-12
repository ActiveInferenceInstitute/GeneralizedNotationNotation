#!/usr/bin/env python3
"""
Test Gnn Validation Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Migrated from test_gnn_core_modules.py
class TestGNNValidation:
    """Test gnn.validation module."""

    @pytest.mark.unit
    def test_validation_imports(self) -> Any:
        """Test that validation module can be imported."""
        from gnn import validate_gnn_structure

        assert callable(validate_gnn_structure)

    @pytest.mark.unit
    def test_validate_gnn_structure_basic(self, sample_gnn_files: Any) -> Any:
        """Test GNN structure validation."""
        from gnn import validate_gnn_structure

        # Get sample file content
        sample_file = list(sample_gnn_files.values())[0]
        content = sample_file.read_text()

        result = validate_gnn_structure(content)

        # Verify validation result structure
        assert isinstance(result, dict)


# Migrated from test_gnn_core_modules.py
class TestGNNSimpleValidator:
    """Test gnn.simple_validator module."""

    @pytest.mark.unit
    def test_simple_validator_imports(self) -> Any:
        """Test that simple validator can be imported."""
        from gnn import simple_validator

        assert hasattr(simple_validator, "SimpleValidator")

    @pytest.mark.unit
    def test_simple_validator_instantiation(self) -> Any:
        """Test SimpleValidator instantiation."""
        from gnn.simple_validator import SimpleValidator

        validator = SimpleValidator()

        # Verify validator has expected methods
        assert hasattr(validator, "validate_file")
        assert hasattr(validator, "validate_directory")

    @pytest.mark.unit
    def test_simple_validation(self, sample_gnn_files: Any) -> Any:
        """Test simple validation functionality."""
        from gnn.simple_validator import SimpleValidator

        validator = SimpleValidator()

        # Get sample file content
        sample_file = list(sample_gnn_files.values())[0]

        result = validator.validate_file(sample_file)

        # Verify validation result
        assert isinstance(result, (dict, bool))
