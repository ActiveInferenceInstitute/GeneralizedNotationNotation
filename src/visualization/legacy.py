#!/usr/bin/env python3
"""
Visualization legacy module for GNN Processing Pipeline.

This module provides legacy function compatibility wrappers.
"""

def matrix_visualizer(*args, **kwargs):
    """Legacy function name compatibility for matrix visualization."""
    from .processor import process_matrix_visualization
    return process_matrix_visualization(*args, **kwargs)

def generate_visualizations(*args, **kwargs):
    """Legacy function name compatibility for visualization generation."""
    from .processor import process_visualization
    return process_visualization(*args, **kwargs)
