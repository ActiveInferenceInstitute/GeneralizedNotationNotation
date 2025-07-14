"""
JAX renderer for GNN specifications.

This module provides comprehensive JAX implementations for POMDPs and other
Active Inference models, including optimized belief updates, value iteration,
and policy optimization using JAX's advanced features.
"""

from .jax_renderer import (
    render_gnn_to_jax,
    render_gnn_to_jax_pomdp,
    render_gnn_to_jax_combined
)

__all__ = [
    'render_gnn_to_jax',
    'render_gnn_to_jax_pomdp', 
    'render_gnn_to_jax_combined'
] 