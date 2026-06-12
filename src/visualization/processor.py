#!/usr/bin/env python3
"""
Visualization processor — re-exports core orchestration and plotting helpers.

Implementation: visualization.core.process, visualization.plotting, visualization.parse.
"""

from typing import Any

from .core.process import process_single_gnn_file, process_visualization
from .matrix.compat import (
    generate_matrix_visualizations,
    parse_matrix_data,
)
from .parse.markdown import parse_gnn_content
from .plotting.utils import (
    safe_tight_layout,
    save_plot_safely,
)

__all__: list[Any] = [
    "process_visualization",
    "process_single_gnn_file",
    "parse_gnn_content",
    "parse_matrix_data",
    "generate_matrix_visualizations",
    "save_plot_safely",
    "safe_tight_layout",
]
