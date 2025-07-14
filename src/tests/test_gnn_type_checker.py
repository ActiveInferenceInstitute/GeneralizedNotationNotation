#!/usr/bin/env python3
"""
Tests for GNN Type Checker

This module tests the GNN type checker functionality, including file validation,
syntax checking, and report generation.
"""

import pytest
import tempfile
from pathlib import Path
from type_checker.checker import GNNTypeChecker, check_gnn_file, validate_syntax

class TestGNNTypeChecker:
    """Test cases for GNNTypeChecker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = GNNTypeChecker(strict_mode=False)
    
    def test_checker_initialization(self):
        """Test that the checker initializes correctly."""
        assert self.checker is not None
        assert hasattr(self.checker, 'check_file')
        assert hasattr(self.checker, 'check_directory')
        assert hasattr(self.checker, 'generate_report')
    
    def test_check_valid_file(self):
        """Test checking a valid GNN file."""
        valid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: TestModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable
y[3,1,type=float]        # Hidden variable

# Connections
Connections:
x - y

# Time
Time:
Static

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(valid_content)
            temp_file = f.name
        
        try:
            # Fix: The API returns 4 values, not 3
            is_valid, errors, warnings, details = self.checker.check_file(temp_file)
            assert is_valid is True
            assert len(errors) == 0
            assert isinstance(warnings, list)
            assert isinstance(details, dict)
        finally:
            Path(temp_file).unlink()
    
    def test_check_invalid_file(self):
        """Test checking an invalid GNN file."""
        invalid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: TestModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable

# Connections
Connections:
x - y                    # y is not defined

# Time
Time:
InvalidTimeSpec        # Invalid time specification

# Equations
Equations:
z = f(y)               # Invalid equation with undefined variables

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(invalid_content)
            temp_file = f.name
        
        try:
            # Fix: The API returns 4 values, not 3
            is_valid, errors, warnings, details = self.checker.check_file(temp_file)
            assert is_valid is False
            assert len(errors) > 0
            assert isinstance(warnings, list)
            assert isinstance(details, dict)
        finally:
            Path(temp_file).unlink()
    
    def test_check_directory(self):
        """Test checking a directory of GNN files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid file
            valid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: ValidModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable
y[3,1,type=float]        # Hidden variable

# Connections
Connections:
x - y

# Time
Time:
Static

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
            valid_file = temp_path / "valid.md"
            valid_file.write_text(valid_content)
            
            # Create invalid file
            invalid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: InvalidModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable

# Connections
Connections:
x - y                    # y is not defined

# Time
Time:
InvalidTimeSpec

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
            invalid_file = temp_path / "invalid.md"
            invalid_file.write_text(invalid_content)
            
            # Test directory checking
            results = self.checker.check_directory(temp_dir)
            assert len(results) == 2
            assert "valid.md" in str(list(results.keys())[0])
            assert "invalid.md" in str(list(results.keys())[1])
    
    def test_generate_report(self):
        """Test report generation."""
        # Create test results
        results = {
            "file1.md": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Warning 1"],
                "details": {"file_path": "file1.md"}
            },
            "file2.md": {
                "is_valid": False,
                "errors": ["Error 1", "Error 2"],
                "warnings": [],
                "details": {"file_path": "file2.md"}
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = self.checker.generate_report(
                results, 
                Path(temp_dir), 
                "test_report.md"
            )
            
            assert Path(report_path).exists()
            report_content = Path(report_path).read_text()
            
            # Check that report contains expected content
            assert "GNN Type Checker Report" in report_content
            assert "Total Files Checked: 2" in report_content
            assert "Valid Files: 1" in report_content
            assert "Invalid Files: 1" in report_content
    
    def test_strict_mode(self):
        """Test strict mode validation."""
        strict_checker = GNNTypeChecker(strict_mode=True)
        
        # Content that would pass in non-strict mode but fail in strict mode
        content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: TestModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable

# Connections
Connections:
x - y

# Time
Time:
Dynamic                 # Dynamic without time specification

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            # Fix: The API returns 4 values, not 3
            is_valid, errors, warnings, details = strict_checker.check_file(temp_file)
            # In strict mode, this should fail due to missing time specification
            assert is_valid is False
            assert len(errors) > 0
        finally:
            Path(temp_file).unlink()

class TestTypeCheckerFunctions:
    """Test cases for standalone type checker functions."""
    
    def test_check_gnn_file_function(self):
        """Test the check_gnn_file function."""
        valid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: TestModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable
y[3,1,type=float]        # Hidden variable

# Connections
Connections:
x - y

# Time
Time:
Static

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(valid_content)
            temp_file = f.name
        
        try:
            result = check_gnn_file(temp_file, strict_mode=False)
            assert result.is_valid is True
            assert len(result.errors) == 0
            assert isinstance(result.warnings, list)
            assert isinstance(result.details, dict)
        finally:
            Path(temp_file).unlink()
    
    def test_validate_syntax_function(self):
        """Test the validate_syntax function."""
        valid_content = """
# GNN Version and Flags
GNNVersionAndFlags: v1.0

# Model Name
ModelName: TestModel

# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable
y[3,1,type=float]        # Hidden variable

# Connections
Connections:
x - y

# Time
Time:
Static

# Footer
Footer: Test footer

# Signature
Signature: Test signature
"""
        
        result = validate_syntax(valid_content, strict_mode=False)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert isinstance(result.warnings, list)
        assert isinstance(result.details, dict)
    
    def test_estimate_resources_function(self):
        """Test the estimate_resources function."""
        from type_checker.checker import estimate_resources
        
        content = """
# State Space Block
StateSpaceBlock:
x[2,1,type=float]        # Observable variable
y[3,1,type=float]        # Hidden variable
z[4,4,type=float]        # Large matrix

# Connections
Connections:
x - y
y - z
"""
        
        resources = estimate_resources(content)
        assert isinstance(resources, dict)
        assert 'total_elements' in resources
        assert 'memory_estimate_mb' in resources
        assert 'computation_complexity' in resources 