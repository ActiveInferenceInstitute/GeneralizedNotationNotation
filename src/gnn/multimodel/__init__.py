"""Multi-model GNN file handling: splitting, per-model parsing, inter-model dependency graphs."""

from gnn.multimodel.dep_graph import (
    DependencyGraph,
    ModelEdge,
    ModelNode,
    build_dependency_graph,
    render_graph_from_file,
)
from gnn.multimodel.multimodel import parse_multimodel, split_models

__all__ = [
    "DependencyGraph",
    "ModelEdge",
    "ModelNode",
    "build_dependency_graph",
    "parse_multimodel",
    "render_graph_from_file",
    "split_models",
]
