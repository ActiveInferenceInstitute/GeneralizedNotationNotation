#!/usr/bin/env python3
"""
Visualization processor — re-exports core orchestration and plotting helpers.

Implementation: visualization.core.process, visualization.plotting, visualization.parse.
"""

from typing import Any

# Imported after ``.core.process``: core.process itself pulls in
# analysis.combined_analysis during package init, so by this line the
# analysis package is fully loaded (import-order-sensitive cycle).
from .analysis import (
    generate_combined_analysis,
    generate_combined_visualizations,
)
from .core.process import (
    discover_visualization_files,
    process_single_gnn_file,
    process_visualization,
)
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
    "discover_visualization_files",
    "process_single_gnn_file",
    "parse_gnn_content",
    "parse_matrix_data",
    "generate_matrix_visualizations",
    "save_plot_safely",
    "safe_tight_layout",
]
