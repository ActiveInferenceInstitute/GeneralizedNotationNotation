"""
Visualization module for GNN Processing Pipeline.

This module provides comprehensive visualization capabilities for GNN files,
including matrix visualizations, network graphs, and combined analysis plots.

The package-root surface re-exports the full documented public API so callers
can import every entry point from ``visualization`` directly:

>>> from visualization import process_visualization, MatrixVisualizer
>>> from visualization import generate_network_visualizations
>>> from visualization import load_visualization_model, GNNParser
"""

from typing import Any

FEATURES: dict[str, Any] = {
    "matrix_visualization": True,
    "network_graphs": True,
    "combined_analysis": True,
    "interactive_plots": True,
    "mcp_integration": True,
}

# Phase 6: numpy and visualization submodules are required core deps per
# pyproject.toml. Unconditional imports — any failure is a real bug.
from .backends import backend_status
from .core.parsed_model import load_visualization_model
from .core.sampling import sample_parsed_data
from .graph import (
    generate_network_visualizations,
    generate_variable_parameter_bipartite,
)
from .graph.stats import compute_connection_statistics
from .matrix.extract import collect_visualization_matrices
from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
from .ontology_visualizer import OntologyVisualizer
from .parse import GNNParser, parse_gnn_content
from .processor import (
    discover_visualization_files,
    generate_combined_analysis,
    generate_combined_visualizations,
    generate_matrix_visualizations,
    parse_matrix_data,
    process_single_gnn_file,
    process_visualization,
)
from .visualizer import (
    GNNVisualizer,
    generate_graph_visualization,
    generate_matrix_visualization,
    generate_visualizations,
)

# Backwards-compatible alias for the pinned package-root statistics helper.
_generate_network_statistics = compute_connection_statistics

__version__ = "1.6.0"


def get_module_info() -> dict:
    """Return visualization module metadata for composability and MCP discovery."""
    return {
        "version": __version__,
        "description": "Visualization utilities for matrices, graphs, and ontology.",
        "features": FEATURES,
        "visualization_types": ["matrix", "graph", "ontology"],
        "backends": backend_status(),
    }


def get_visualization_options() -> dict:
    """Return available visualization configuration options."""
    return {
        "matrix_types": ["heatmap", "statistics", "tensor", "pomdp_analysis"],
        "graph_types": ["connections", "bipartite", "combined"],
        "output_formats": ["png", "json", "html", "viz_manifest"],
    }


__all__: list[Any] = [
    "MatrixVisualizer",
    "GNNVisualizer",
    "OntologyVisualizer",
    "GNNParser",
    "process_matrix_visualization",
    "process_visualization",
    "discover_visualization_files",
    "process_single_gnn_file",
    "generate_graph_visualization",
    "generate_matrix_visualization",
    "generate_matrix_visualizations",
    "generate_network_visualizations",
    "generate_variable_parameter_bipartite",
    "generate_combined_analysis",
    "generate_combined_visualizations",
    "generate_visualizations",
    "parse_gnn_content",
    "parse_matrix_data",
    "load_visualization_model",
    "sample_parsed_data",
    "collect_visualization_matrices",
    "compute_connection_statistics",
    "backend_status",
    "__version__",
]


# Re-exported for the test that pins the package-root statistics helper.
__all__.append("_generate_network_statistics")
