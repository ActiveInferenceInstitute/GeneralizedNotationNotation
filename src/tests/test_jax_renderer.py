#!/usr/bin/env python3
"""
Tests for render/jax/jax_renderer.py — pure Python parsing functions
that don't require JAX to be installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")


@pytest.fixture(scope="module")
def mod():
    try:
        import render.jax.jax_renderer as m
        return m
    except ImportError:
        pytest.skip("render.jax.jax_renderer not importable")


class TestParseGnnMatrixString:
    def test_3x3_matrix_with_braces(self, mod):
        matrix = mod._parse_gnn_matrix_string("{(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)}")
        assert matrix.shape == (3, 3)
        assert abs(matrix[0, 0] - 0.9) < 1e-6

    def test_returns_numpy_array(self, mod):
        result = mod._parse_gnn_matrix_string("{(0.5,0.5)}")
        assert isinstance(result, np.ndarray)

    def test_invalid_returns_fallback_array(self, mod):
        result = mod._parse_gnn_matrix_string("NOT_A_MATRIX")
        assert isinstance(result, np.ndarray)


class TestParseMatrixString:
    def test_semicolon_separated_rows(self, mod):
        matrix = mod._parse_matrix_string("0.9,0.1;0.1,0.9")
        assert matrix.shape == (2, 2)
        assert abs(matrix[0, 0] - 0.9) < 1e-6

    def test_single_row(self, mod):
        matrix = mod._parse_matrix_string("0.5,0.3,0.2")
        assert matrix.ndim >= 1
        assert matrix.shape[-1] == 3

    def test_empty_string_returns_array(self, mod):
        result = mod._parse_matrix_string("")
        assert isinstance(result, np.ndarray)


class TestParseVectorString:
    def test_basic_vector(self, mod):
        vec = mod._parse_vector_string("0.5,0.3,0.2")
        assert vec.shape == (3,)
        assert abs(vec.sum() - 1.0) < 1e-6

    def test_returns_numpy_array(self, mod):
        result = mod._parse_vector_string("0.5,0.5")
        assert isinstance(result, np.ndarray)

    def test_invalid_falls_back(self, mod):
        result = mod._parse_vector_string("INVALID")
        assert isinstance(result, np.ndarray)


class TestExtractGnnMatrices:
    def test_empty_spec_returns_dict(self, mod):
        result = mod._extract_gnn_matrices({})
        assert isinstance(result, dict)

    def test_spec_with_parameters_returns_dict(self, mod):
        spec = {
            "parameters": {"A": "{(0.9,0.1),(0.1,0.9)}"},
            "state_space": {"A": {"dimensions": [2, 2], "type": "float"}},
        }
        result = mod._extract_gnn_matrices(spec)
        assert isinstance(result, dict)
