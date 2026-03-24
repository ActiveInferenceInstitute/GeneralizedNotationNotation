"""
RxInfer rendering module for GNN.
"""

try:
    from .rxinfer_renderer import render_gnn_to_rxinfer
    from .toml_generator import render_gnn_to_rxinfer_toml
    __all__ = ["render_gnn_to_rxinfer_toml", "render_gnn_to_rxinfer"]
except ImportError:
    # toml not available, but core RxInfer generation still works
    try:
        from .rxinfer_renderer import render_gnn_to_rxinfer
        __all__ = ["render_gnn_to_rxinfer"]
    except ImportError:
        __all__ = []
