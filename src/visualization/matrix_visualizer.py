"""Facade: matrix implementation lives in visualization.matrix."""

from .matrix import (
    MatrixVisualizer,
    generate_matrix_visualizations,
    process_matrix_visualization,
)

__all__ = [
    "MatrixVisualizer",
    "generate_matrix_visualizations",
    "process_matrix_visualization",
]
