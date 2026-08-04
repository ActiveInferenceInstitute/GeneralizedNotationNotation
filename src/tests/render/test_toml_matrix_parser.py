#!/usr/bin/env python3
"""
Focused unit tests for the GNN parenthesized-tuple matrix parsers in
src/render/rxinfer/toml_generator.py.

These tests pin the correct parsing behavior for the parenthesized-tuple matrix
notation used in the exemplar GNN files, and guard against the historical
silent-correctness footgun where malformed input was silently replaced with an
identity matrix. Malformed input must now raise a clear ValueError instead.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from render.rxinfer.toml_generator import (  # noqa: E402
    _parse_gnn_3d_matrix,
    _parse_gnn_matrix,
    _parse_gnn_vector,
)


class TestParseGnnMatrix:
    """Tests for _parse_gnn_matrix (2D parenthesized-tuple matrices)."""

    def test_valid_single_line_fully_parenthesized(self) -> Any:
        """'((a,b,c),(d,e,f),(g,h,i))' form parses into a list of lists."""
        result = _parse_gnn_matrix(
            "((0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9))"
        )
        assert result == [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    def test_valid_brace_wrapped_single_line(self) -> Any:
        """'{(a,b,c),(d,e,f),(g,h,i)}' form parses identically to the bare form."""
        result = _parse_gnn_matrix(
            "{(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)}"
        )
        assert result == [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    def test_valid_multiline_brace_exemplar_form(self) -> Any:
        """Multi-line brace form used by the exemplar A matrices parses."""
        result = _parse_gnn_matrix(
            "{\n"
            "  (1.0, 0.0, 0.0, 0.0),\n"
            "  (0.0, 1.0, 0.0, 0.0),\n"
            "  (0.0, 0.0, 1.0, 0.0),\n"
            "  (0.0, 0.0, 0.0, 1.0)\n"
            "}"
        )
        assert len(result) == 4
        assert all(len(row) == 4 for row in result)
        assert result[0] == [1.0, 0.0, 0.0, 0.0]
        assert result[3] == [0.0, 0.0, 0.0, 1.0]

    @pytest.mark.parametrize(
        "malformed",
        [
            "NOT_A_MATRIX",
            "(a,b,c)",  # non-numeric element
            "",  # empty
            "((1.0,2.0),(1.0))",  # ragged rows
            "((1.0,2.0),(3.0,4.0),(5.0))",  # inconsistent row length
        ],
    )
    def test_malformed_raises_not_identity(self, malformed: str) -> Any:
        """Genuinely malformed input raises ValueError (never identity fallback)."""
        with pytest.raises(ValueError):
            _parse_gnn_matrix(malformed)

    def test_malformed_never_returns_identity(self) -> Any:
        """Guard: malformed input must not silently return a 3x3 identity."""
        with pytest.raises(ValueError):
            _parse_gnn_matrix("((1.0,2.0),(3.0,4.0))xyz")


class TestParseGnn3dMatrix:
    """Tests for _parse_gnn_3d_matrix (3D B transition tensor)."""

    def test_valid_nested_b_form(self) -> Any:
        """Nested parenthesized B form parses to a list of 2D matrices."""
        b = _parse_gnn_3d_matrix(
            "{( (0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9) ), "
            "( (0.05,0.9,0.05), (0.9,0.05,0.05), (0.05,0.05,0.9) )}"
        )
        assert len(b) == 2  # two actions
        assert b[0] == [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]
        assert b[1] == [
            [0.05, 0.9, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.05, 0.9],
        ]

    def test_valid_exemplar_multiline_b(self) -> Any:
        """Multi-line exemplar B form parses to the right 3D shape."""
        b = _parse_gnn_3d_matrix(
            "{\n"
            "  ( (0.9, 0.1, 0.0, 0.0), (0.1, 0.9, 0.0, 0.0), "
            "(0.0, 0.0, 0.9, 0.1), (0.0, 0.0, 0.1, 0.9) ),\n"
            "  ( (0.1, 0.9, 0.0, 0.0), (0.9, 0.1, 0.0, 0.0), "
            "(0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.9, 0.1) )\n"
            "}"
        )
        assert len(b) == 2
        assert all(len(m) == 4 for m in b)
        assert all(all(len(row) == 4 for row in m) for m in b)
        assert b[0][0] == [0.9, 0.1, 0.0, 0.0]
        assert b[1][3] == [0.0, 0.0, 0.9, 0.1]

    def test_malformed_raises_not_identity(self) -> Any:
        """Malformed 3D input raises ValueError, never the identity fallback."""
        with pytest.raises(ValueError):
            _parse_gnn_3d_matrix("NOT_A_3D_MATRIX")


class TestParseGnnVector:
    """Tests for _parse_gnn_vector."""

    def test_valid_vector(self) -> Any:
        result = _parse_gnn_vector("{(0.0, 0.0, 0.0, 3.0)}")
        assert result == [0.0, 0.0, 0.0, 3.0]

    def test_valid_bare_vector(self) -> Any:
        result = _parse_gnn_vector("(0.25, 0.25, 0.25, 0.25)")
        assert result == [0.25, 0.25, 0.25, 0.25]

    def test_malformed_raises_not_uniform(self) -> Any:
        """Malformed vector raises ValueError instead of returning a uniform vector."""
        with pytest.raises(ValueError):
            _parse_gnn_vector("NOT_A_VECTOR")
