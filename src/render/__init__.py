"""
Rendering module for GNN specifications

This module contains the functionality to render GNN specifications into 
executable formats for various target platforms.
"""

# Main entry point
from .render import render_gnn_spec, main

# Target-specific renderers
from .pymdp_renderer import render_gnn_to_pymdp
from .rxinfer import render_gnn_to_rxinfer_jl

__all__ = [
    'render_gnn_spec',  # Main render function
    'main',             # CLI entry point
    'render_gnn_to_pymdp',  # PyMDP-specific renderer
    'render_gnn_to_rxinfer_jl'  # RxInfer-specific renderer
] 