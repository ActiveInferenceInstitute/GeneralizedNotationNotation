"""
Renderers for GNN specifications to various target languages and frameworks.

This package contains modules for rendering GNN specifications to:
- RxInfer.jl
- PyMDP
- Other simulators
"""

# Import renderers here to make them available at package level
# Changed import to use only render_gnn_to_rxinfer_toml
from .rxinfer import render_gnn_to_rxinfer_toml

from .render import render_gnn_spec, main

# Target-specific renderers
from .pymdp_renderer import render_gnn_to_pymdp

__all__ = [
    'render_gnn_spec',  # Main render function
    'main',             # CLI entry point
    'render_gnn_to_pymdp',  # PyMDP-specific renderer
    'render_gnn_to_rxinfer_toml'  # RxInfer-specific renderer
] 