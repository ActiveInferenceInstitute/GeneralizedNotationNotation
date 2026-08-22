"""Tests for visualization/matrix_compat.py facade.

The facade is a thin re-export forwarding to visualization.matrix.compat.
These tests verify the forwarding surface resolves and behaves correctly
without duplicating the matrix module's own coverage.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMatrixCompatFacade:
    """Test the facade re-exports resolve to the real implementations."""

    def test_parse_matrix_data_is_callable(self) -> None:
        from visualization.matrix_compat import parse_matrix_data

        assert callable(parse_matrix_data)

    def test_parse_matrix_data_parses_string(self) -> None:
        import numpy as np

        from visualization.matrix_compat import parse_matrix_data

        # Real string-matrix parse forwards to numpy.
        result = parse_matrix_data("[[1 2]\n [3 4]]")
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size == 4

    def test_generate_matrix_visualizations_is_callable(self) -> None:
        from visualization.matrix_compat import generate_matrix_visualizations

        assert callable(generate_matrix_visualizations)

    def test_all_matches_public_surface(self) -> None:
        import visualization.matrix_compat as facade

        assert facade.__all__ == [
            "parse_matrix_data",
            "generate_matrix_visualizations",
        ]
        for name in facade.__all__:
            assert hasattr(facade, name)
            assert callable(getattr(facade, name))

    def test_forwarding_identity(self) -> None:
        from visualization.matrix import compat as real
        from visualization.matrix_compat import (
            generate_matrix_visualizations,
            parse_matrix_data,
        )

        # The facade must forward the exact same callables.
        assert parse_matrix_data is real.parse_matrix_data
        assert generate_matrix_visualizations is real.generate_matrix_visualizations