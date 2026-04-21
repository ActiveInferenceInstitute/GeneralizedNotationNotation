"""
DisCoPy renderer module for GNN specifications.

Exposes a single entry point, ``render_gnn_to_discopy``, that emits a runnable
Python module containing the DisCoPy categorical diagram for the supplied GNN
spec. A JAX-backed variant was historically re-exported from this package but
never existed in ``discopy_renderer`` — it has been removed to match reality.
"""

from .discopy_renderer import render_gnn_to_discopy

__all__ = ["render_gnn_to_discopy"]
