"""
Advanced Visualization Module for GNN Processing Pipeline

This module provides advanced visualization capabilities for GNN models,
including interactive dashboards, 3D visualizations, and exploration tools.
"""

from .visualizer import AdvancedVisualizer, process_advanced_visualization
from .dashboard import DashboardGenerator, generate_dashboard
from .html_generator import HTMLVisualizationGenerator
from .data_extractor import VisualizationDataExtractor

# Export the missing functions that scripts are looking for
def visualizer(*args, **kwargs):
    """Legacy function name compatibility for advanced visualization."""
    return process_advanced_visualization(*args, **kwargs)

def dashboard(*args, **kwargs):
    """Legacy function name compatibility for dashboard generation."""
    return generate_dashboard(*args, **kwargs)

__all__ = [
    'AdvancedVisualizer',
    'DashboardGenerator', 
    'HTMLVisualizationGenerator',
    'VisualizationDataExtractor',
    'visualizer',
    'dashboard',
    'process_advanced_visualization',
    'generate_dashboard'
]
