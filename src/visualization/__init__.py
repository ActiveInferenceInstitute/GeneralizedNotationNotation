"""
Visualization module for GNN Processing Pipeline.

This module provides comprehensive visualization capabilities for GNN files,
including matrix visualizations, network graphs, and combined analysis plots.
"""

# Import main classes
from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
from .visualizer import GNNVisualizer
from .ontology_visualizer import OntologyVisualizer

# Import processor functions
from .processor import (
    process_visualization,
    process_single_gnn_file,
    parse_gnn_content,
    parse_matrix_data,
    generate_matrix_visualizations,
    generate_network_visualizations,
    generate_combined_analysis,
    generate_combined_visualizations
)

# Import legacy compatibility functions
from .legacy import (
    matrix_visualizer,
    generate_visualizations
)

# Add to __all__ for proper exports
__all__ = [
    'MatrixVisualizer', 'GNNVisualizer', 'OntologyVisualizer',
    'matrix_visualizer', 'process_matrix_visualization', 'process_visualization',
    'generate_visualizations'
]
