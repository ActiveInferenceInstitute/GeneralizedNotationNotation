"""
RxInfer rendering module for GNN.
"""

try:
    from .toml_generator import render_gnn_to_rxinfer_toml
    __all__ = ["render_gnn_to_rxinfer_toml"]
except ImportError:
    # toml not available, but core RxInfer generation still works
    __all__ = [] 