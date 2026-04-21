"""
Visualization module for GNN Processing Pipeline.

This module provides comprehensive visualization capabilities for GNN files,
including matrix visualizations, network graphs, and combined analysis plots.
"""

FEATURES = {
    "matrix_visualization": True,
    "network_graphs": True,
    "combined_analysis": True,
    "interactive_plots": True,
    "mcp_integration": True
}

# Typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Phase 6: numpy and visualization submodules are required core deps per
# pyproject.toml. Unconditional imports — any failure is a real bug.
import numpy as np

from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
from .visualizer import (
    GNNVisualizer,
    generate_graph_visualization,
    generate_matrix_visualization,
    generate_visualizations,
)
from .ontology_visualizer import OntologyVisualizer

# GraphVisualizer is a historical alias — GNNVisualizer is the canonical class.
GraphVisualizer = GNNVisualizer

# Import processor functions
from .processor import (
    generate_matrix_visualizations,
    parse_gnn_content,
    parse_matrix_data,
    process_single_gnn_file,
    process_visualization,
)

__version__ = "1.6.0"

def get_module_info() -> dict:
    """Return visualization module metadata for composability and MCP discovery."""
    return {
        "version": __version__,
        "description": "Visualization utilities for matrices, graphs, and ontology.",
        "features": FEATURES,
        "visualization_types": ["matrix", "graph", "ontology"]
    }

def get_visualization_options() -> dict:
    """Return available visualization configuration options."""
    return {
        "matrix_types": ["heatmap", "statistics", "tensor", "pomdp_analysis"],
        "graph_types": ["connections", "bipartite", "combined"],
        "output_formats": ["png", "json", "html", "viz_manifest"],
    }


def _generate_network_statistics(variables: Dict[str, Any], connections: List[Dict]) -> Dict[str, Any]:
        """Generate network statistics from variables and connections."""
        node_degrees = {}
        for conn in connections:
            source = conn.get("source", "unknown")
            target = conn.get("target", "unknown")
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1

        if node_degrees:
            degrees = list(node_degrees.values())
            stats = {
                "total_nodes": len(variables),
                "total_connections": len(connections),
                "average_degree": sum(degrees) / len(degrees),
                "max_degree": max(degrees),
                "min_degree": min(degrees),
                "node_degree_distribution": node_degrees,
                "isolated_nodes": len([v for v in variables.keys() if v not in node_degrees]),
                "hub_nodes": [node for node, degree in node_degrees.items() if degree > 2]
            }
        else:
            stats = {
                "total_nodes": len(variables),
                "total_connections": len(connections),
                "average_degree": 0,
                "max_degree": 0,
                "min_degree": 0,
                "node_degree_distribution": {},
                "isolated_nodes": len(variables),
                "hub_nodes": []
            }

        return stats

__all__ = [
    'MatrixVisualizer', 'GNNVisualizer', 'OntologyVisualizer', 'GraphVisualizer',
    'process_matrix_visualization', 'process_visualization',
    'generate_graph_visualization', 'generate_matrix_visualization', 'generate_visualizations',
    '__version__'
]
