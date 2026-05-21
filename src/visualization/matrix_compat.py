"""Facade: matrix helpers live in visualization.matrix.compat."""

from typing import Any

from .matrix.compat import (
    generate_matrix_visualizations,
    parse_matrix_data,
)

__all__: list[Any] = ["parse_matrix_data", "generate_matrix_visualizations"]
