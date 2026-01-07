#!/usr/bin/env python3
"""
Unit tests for ActiveInference.jl renderer helper functions.

Focuses on ensuring robust conversion of various Python data structures (Lists, Tuples, Nested)
into valid Julia matrix syntax.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the internal helper function
# Since it is now at module level, we can import it (it is private but testable)
from render.activeinference_jl.activeinference_renderer import _matrix_to_julia

class TestMatrixToJulia:
    """Test the _matrix_to_julia function for various input types."""

    def test_basic_vector(self):
        """Test converting a simple 1D list/vector."""
        data = [0.1, 0.9]
        result = _matrix_to_julia(data)
        assert result == "[0.1, 0.9]"
        
        # Test tuple version
        data_tuple = (0.1, 0.9)
        result_tuple = _matrix_to_julia(data_tuple)
        assert result_tuple == "[0.1, 0.9]"

    def test_2d_matrix_list_of_lists(self):
        """Test converting 2D list of lists (canonical A matrix)."""
        data = [[0.9, 0.1], [0.1, 0.9]]
        result = _matrix_to_julia(data)
        # Julia matrix syntax: [row1; row2; ...]
        # Space separated elements in row
        # Expected: "[0.9 0.1; 0.1 0.9]"
        assert result == "[0.9 0.1; 0.1 0.9]" or result == "[0.9, 0.1; 0.1, 0.9]" or "0.9 0.1" in result

    def test_2d_matrix_tuple_of_lists(self):
        """Test converting Tuple of Lists (The bug case)."""
        # This was causing failures before: 
        # ([0.9, 0.1], [0.1, 0.9]) was treated as a 1D vector of lists -> [[0.9,0.1], [0.1,0.9]] 
        # instead of a proper 2D matrix.
        data = ([0.9, 0.1], [0.1, 0.9])
        result = _matrix_to_julia(data)
        # Should be treated same as list of lists
        assert ";" in result
        assert result.startswith("[")
        assert result.endswith("]")
        # Should NOT be "[ [0.9, 0.1], [0.1, 0.9] ]"
        assert result == "[0.9 0.1; 0.1 0.9]"

    def test_2d_matrix_tuple_of_tuples(self):
        """Test converting Tuple of Tuples."""
        data = ((0.9, 0.1), (0.1, 0.9))
        result = _matrix_to_julia(data)
        assert result == "[0.9 0.1; 0.1 0.9]"

    def test_3d_matrix_list_of_list_of_lists(self):
        """Test converting 3D tensor (B matrix)."""
        # B[next_state, current_state, action]
        # shape (2, 2, 2)
        slice1 = [[1.0, 0.0], [0.0, 1.0]] # Identity
        slice2 = [[0.0, 1.0], [1.0, 0.0]] # Flip
        data = [slice1, slice2]
        
        result = _matrix_to_julia(data)
        # format: cat([slice1], [slice2]; dims=3)
        assert result.startswith("cat(")
        assert "dims=3" in result
        assert "[1.0 0.0; 0.0 1.0]" in result
        assert "[0.0 1.0; 1.0 0.0]" in result

    def test_3d_matrix_nested_tuples(self):
        """Test converting 3D tensor with mixed/nested tuples."""
        slice1 = ((1.0, 0.0), (0.0, 1.0))
        slice2 = ((0.0, 1.0), (1.0, 0.0))
        data = (slice1, slice2)
        
        result = _matrix_to_julia(data)
        assert result.startswith("cat(")
        assert "[1.0 0.0; 0.0 1.0]" in result
        
    def test_string_input(self):
        """Test string fallback behavior."""
        data = "[[0.9, 0.1], [0.1, 0.9]]"
        # The function attempts literal_eval for strings
        result = _matrix_to_julia(data)
        assert result == "[0.9 0.1; 0.1 0.9]"

