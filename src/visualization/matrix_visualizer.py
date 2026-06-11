"""Facade: matrix implementation lives in visualization.matrix."""

from typing import Any

from .matrix import (
    MatrixVisualizer,
    generate_matrix_visualizations,
    process_matrix_visualization,
)

__all__: list[Any] = [
    "MatrixVisualizer",
    "generate_matrix_visualizations",
    "process_matrix_visualization",
]
