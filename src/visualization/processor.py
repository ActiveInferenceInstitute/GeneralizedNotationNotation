#!/usr/bin/env python3
"""
Visualization processor — re-exports core orchestration and plotting helpers.

Implementation: visualization.core.process, visualization.plotting, visualization.parse.
"""

from visualization.core.process import process_single_gnn_file, process_visualization
from visualization.matrix.compat import generate_matrix_visualizations, parse_matrix_data
from visualization.parse.markdown import parse_gnn_content
from visualization.plotting.utils import (
    _safe_tight_layout,
    _save_plot_safely,
    save_plot_safely,
    safe_tight_layout,
)

__all__ = [
    "process_visualization",
    "process_single_gnn_file",
    "parse_gnn_content",
    "parse_matrix_data",
    "generate_matrix_visualizations",
    "save_plot_safely",
    "safe_tight_layout",
    "_save_plot_safely",
    "_safe_tight_layout",
]
