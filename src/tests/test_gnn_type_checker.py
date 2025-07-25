"""
Test suite for GNN Type Checker

All test outputs must go under output/type_check/ or a temp subfolder thereof.
The type checker CLI now enforces this policy and will refuse to run if --output-dir is not a subdirectory named 'type_check'.
"""

import pytest
import os
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Test markers
pytestmark = [pytest.mark.type_checking, pytest.mark.safe_to_fail, pytest.mark.fast]

# Add the src directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.type_checker import GNNTypeChecker


class TestGNNTypeChecker(unittest.TestCase):
    """Tests for the GNNTypeChecker class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.checker = GNNTypeChecker()
        
        # Create a valid GNN file for testing
        self.valid_gnn_content = """# GNN Example: Valid Test Model
# Format: Markdown representation of a Valid Test Model
# Version: 1.0
# This file is machine-readable

## GNNSection
TestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Valid Test Model

## StateSpaceBlock
x[2,1,type=float]      # Observable variable
y[3,1,type=float]      # Hidden variable

## Connections
x-y                    # Bidirectional connection

## InitialParameterization
x={0.0,0.0}            # Initial values for x
y={1.0,1.0,1.0}        # Initial values for y

## Equations
x = f(y)               # Simple equation

## Time
Static

## Footer
Valid Test Model

## Signature
NA
"""
        
        # Create an invalid GNN file for testing
        self.invalid_gnn_content = """# GNN Example: Invalid Test Model
# Format: Markdown representation of an Invalid Test Model
# Version: 1.0
# This file is machine-readable

## GNNSection
TestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Invalid Test Model

## StateSpaceBlock
x[2,1,type=float]      # Observable variable

## Connections
x-y                    # Invalid connection to undefined variable y

## InitialParameterization
x={0.0,0.0}            # Initial values for x

## Equations
z = f(y)               # Invalid equation with undefined variables

## Time
InvalidTimeSpec        # Invalid time specification

## Footer
Invalid Test Model

## Signature
NA
"""
    
    def test_check_valid_file(self):
        """Test checking a valid GNN file."""
        with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(self.valid_gnn_content)
            temp_file = f.name
        
        try:
            result = self.checker.check_file(temp_file)
            is_valid, errors, warnings, metadata = result
            
            self.assertTrue(is_valid, f"Expected valid file to pass checks, but got errors: {errors}")
            self.assertEqual(len(errors), 0, "Expected no errors for valid file")
        finally:
            os.unlink(temp_file)
    
    def test_check_invalid_file(self):
        """Test checking an invalid GNN file."""
        with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(self.invalid_gnn_content)
            temp_file = f.name
        
        try:
            result = self.checker.check_file(temp_file)
            is_valid, errors, warnings, metadata = result
            
            self.assertFalse(is_valid, "Expected invalid file to fail checks")
            self.assertGreater(len(errors), 0, "Expected at least one error for invalid file")
            
            # Check for specific errors
            connection_error = any("Connection references undefined variable: y" in error for error in errors)
            time_error = any("Invalid time specification" in error for error in errors)
            
            self.assertTrue(connection_error, "Expected error about undefined variable in connection")
            self.assertTrue(time_error, "Expected error about invalid time specification")
        finally:
            os.unlink(temp_file)
    
    def test_check_directory(self):
        """Test checking a directory of GNN files."""
        with TemporaryDirectory() as temp_dir:
            # Create a valid file
            valid_path = os.path.join(temp_dir, "valid.md")
            with open(valid_path, 'w') as f:
                f.write(self.valid_gnn_content)
            
            # Create an invalid file
            invalid_path = os.path.join(temp_dir, "invalid.md")
            with open(invalid_path, 'w') as f:
                f.write(self.invalid_gnn_content)
            
            # Check the directory
            results = self.checker.check_directory(temp_dir)
            
            self.assertEqual(len(results), 2, "Expected results for 2 files")
            self.assertTrue(results[valid_path]["is_valid"], "Expected valid file to pass")
            self.assertFalse(results[invalid_path]["is_valid"], "Expected invalid file to fail")
    
    def test_generate_report(self):
        """Test generating a report from check results."""
        results = {
            "file1.md": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Warning 1"]
            },
            "file2.md": {
                "is_valid": False,
                "errors": ["Error 1", "Error 2"],
                "warnings": []
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_file = f.name
        
        try:
            # Get directory and filename from temp_file
            temp_file_path = Path(temp_file)
            output_dir = temp_file_path.parent
            report_filename = temp_file_path.name
            
            report = self.checker.generate_report(results, output_dir_base=output_dir, report_md_filename=report_filename)
            
            # Check that the report was written to the file
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r') as f:
                file_content = f.read()
                self.assertEqual(file_content, report)
            
            # Check report content
            self.assertIn("**Total Files Checked:** 2", report)
            self.assertIn("file1.md: ✅ VALID", report)
            self.assertIn("file2.md: ❌ INVALID", report)
            self.assertIn("Warning 1", report)
            self.assertIn("Error 1", report)
            self.assertIn("Error 2", report)
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    unittest.main() 