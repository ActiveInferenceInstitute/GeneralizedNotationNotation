"""Shim: matrix helpers live in visualization.matrix.compat."""

from visualization.matrix.compat import (
    generate_matrix_visualizations,
    parse_matrix_data,
)

__all__ = ["parse_matrix_data", "generate_matrix_visualizations"]
