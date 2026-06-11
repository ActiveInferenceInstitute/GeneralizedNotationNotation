"""
JAX renderer for GNN specifications.

This module provides comprehensive JAX implementations for POMDPs and other
Active Inference models, including optimized belief updates, value iteration,
and policy optimization using JAX's advanced features.
"""

from typing import Any

from .jax_renderer import (
    render_gnn_to_jax,
    render_gnn_to_jax_combined,
    render_gnn_to_jax_pomdp,
)

__all__: list[Any] = [
    "render_gnn_to_jax",
    "render_gnn_to_jax_pomdp",
    "render_gnn_to_jax_combined",
]
