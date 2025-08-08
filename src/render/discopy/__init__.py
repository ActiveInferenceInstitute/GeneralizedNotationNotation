"""
DisCoPy renderer module for GNN specifications.

This module provides rendering capabilities for GNN specifications to DisCoPy
categorical diagrams and JAX-evaluatable matrix diagrams.
"""

# Base DisCoPy renderer is required
from .discopy_renderer import render_gnn_to_discopy  # type: ignore

# JAX-enhanced renderer is optional; degrade gracefully if unavailable
try:
    from .discopy_renderer import render_gnn_to_discopy_jax  # type: ignore
    _DISCOPY_JAX_AVAILABLE = True
except Exception:
    render_gnn_to_discopy_jax = None  # type: ignore
    _DISCOPY_JAX_AVAILABLE = False

__all__ = ['render_gnn_to_discopy'] + (['render_gnn_to_discopy_jax'] if _DISCOPY_JAX_AVAILABLE else [])