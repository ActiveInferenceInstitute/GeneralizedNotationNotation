"""
ActiveInference.jl renderer module for GNN specifications.

This module provides streamlined rendering capabilities for GNN specifications to
ActiveInference.jl code, focusing on core POMDP functionality.
"""

from .activeinference_renderer import (
    extract_model_info,
    generate_activeinference_script,
    render_gnn_to_activeinference_combined,
    render_gnn_to_activeinference_jl,
)

__all__ = [
    'render_gnn_to_activeinference_jl',
    'render_gnn_to_activeinference_combined',
    'extract_model_info',
    'generate_activeinference_script'
]
