"""
DisCoPy renderer module for GNN specifications.

This module provides rendering capabilities for GNN specifications to DisCoPy
categorical diagrams and JAX-evaluatable matrix diagrams.
"""

from .discopy_renderer import render_gnn_to_discopy, render_gnn_to_discopy_jax

__all__ = [
    'render_gnn_to_discopy',
    'render_gnn_to_discopy_jax'
] 