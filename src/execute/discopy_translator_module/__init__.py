"""
DisCoPy Translator Module for GNN Processing Pipeline

This module provides translation functionality from GNN specifications
to DisCoPy categorical diagrams with JAX evaluation capabilities.
"""

from typing import Any

__version__ = "1.1.3"

# Import main functionality
from .translator import (
    JAX_FULLY_OPERATIONAL,
    MATPLOTLIB_AVAILABLE,
    gnn_file_to_discopy_diagram,
    gnn_file_to_discopy_matrix_diagram,
)
from .visualize_jax_output import plot_tensor_output

__all__: list[Any] = [
    "JAX_FULLY_OPERATIONAL",
    "MATPLOTLIB_AVAILABLE",
    "gnn_file_to_discopy_diagram",
    "gnn_file_to_discopy_matrix_diagram",
    "plot_tensor_output",
]
