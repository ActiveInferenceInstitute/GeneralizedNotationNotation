"""
RxInfer rendering module for GNN.

The canonical renderer is ``rxinfer_renderer.py``, which emits genuine
``@model`` + ``infer()`` Julia code backed by a committed Project.toml
environment. The legacy ``toml_generator.py`` is deprecated.
"""

from typing import Any

try:
    from .rxinfer_renderer import render_gnn_to_rxinfer

    __all__: list[Any] = ["render_gnn_to_rxinfer"]
except ImportError:
    __all__ = []
