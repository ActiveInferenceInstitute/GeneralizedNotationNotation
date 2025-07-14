"""
GNN Visualization Module

This module provides tools for visualizing Generalized Notation Notation (GNN) models
and generating comprehensive state-space visualizations.
"""

from .visualizer import (
    GNNVisualizer,
    generate_graph_visualization,
    generate_matrix_visualization,
    create_visualization_report,
    visualize_gnn_model
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN model visualization and analysis"

# Feature availability flags
FEATURES = {
    'graph_visualization': True,
    'matrix_visualization': True,
    'ontology_visualization': True,
    'state_space_visualization': True,
    'connection_visualization': True,
    'report_generation': True
}

# Main API functions
__all__ = [
    'GNNVisualizer',
    'generate_graph_visualization',
    'generate_matrix_visualization',
    'create_visualization_report',
    'visualize_gnn_model',
    'FEATURES',
    '__version__'
]


def get_module_info():
    """Get comprehensive information about the visualization module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'visualization_types': [],
        'output_formats': []
    }
    
    # Visualization types
    info['visualization_types'].extend([
        'State space diagrams',
        'Connection graphs',
        'Matrix heatmaps',
        'Ontology relationship diagrams',
        'Combined model views',
        'Text-based summaries'
    ])
    
    # Output formats
    info['output_formats'].extend(['PNG', 'SVG', 'PDF', 'HTML', 'JSON'])
    
    return info


def get_visualization_options() -> dict:
    """Get information about available visualization options."""
    return {
        'graph_types': {
            'network': 'Network graph visualization',
            'hierarchical': 'Hierarchical tree visualization',
            'circular': 'Circular layout visualization',
            'force_directed': 'Force-directed graph layout'
        },
        'matrix_types': {
            'heatmap': 'Matrix heatmap visualization',
            'sparse': 'Sparse matrix visualization',
            'correlation': 'Correlation matrix visualization'
        },
        'color_schemes': {
            'viridis': 'Viridis color scheme',
            'plasma': 'Plasma color scheme',
            'inferno': 'Inferno color scheme',
            'magma': 'Magma color scheme'
        },
        'output_options': {
            'high_resolution': 'High resolution output',
            'interactive': 'Interactive visualizations',
            'animated': 'Animated visualizations'
        }
    } 