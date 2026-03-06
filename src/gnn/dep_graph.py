#!/usr/bin/env python3
"""
Model Dependency Graph — Visualize inter-model connections.

Builds a dependency graph from multi-model GNN files and renders
as Mermaid diagrams or plain-text adjacency lists.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ModelNode:
    """A model in the dependency graph."""
    name: str
    index: int
    variable_count: int = 0
    connection_count: int = 0


@dataclass
class ModelEdge:
    """A dependency between two models."""
    source_model: str
    target_model: str
    shared_variables: List[str] = field(default_factory=list)
    label: str = ""


@dataclass
class DependencyGraph:
    """Graph of inter-model dependencies."""
    nodes: List[ModelNode] = field(default_factory=list)
    edges: List[ModelEdge] = field(default_factory=list)

    def to_mermaid(self) -> str:
        """Render as Mermaid flowchart."""
        lines = ["graph TD"]

        for node in self.nodes:
            label = f"{node.name}\\n({node.variable_count} vars)"
            lines.append(f"    {node.name}[\"{label}\"]")

        for edge in self.edges:
            label_str = f"|{edge.label}|" if edge.label else ""
            lines.append(f"    {edge.source_model} -->{label_str} {edge.target_model}")

        return "\n".join(lines)

    def to_adjacency_list(self) -> str:
        """Render as plain-text adjacency list."""
        lines = ["Dependency Graph:"]
        for node in self.nodes:
            deps = [e.target_model for e in self.edges if e.source_model == node.name]
            if deps:
                lines.append(f"  {node.name} → {', '.join(deps)}")
            else:
                lines.append(f"  {node.name} (leaf)")
        return "\n".join(lines)


def build_dependency_graph(
    models: List[Dict[str, Any]],
    file_path: Optional[str] = None,
) -> DependencyGraph:
    """
    Build dependency graph from parsed multi-model data.

    Detects shared variables between models to infer dependencies.

    Args:
        models: List of parsed model dicts (from multimodel.parse_multimodel).

    Returns:
        DependencyGraph with nodes and edges.
    """
    graph = DependencyGraph()

    # Create nodes
    for i, model in enumerate(models):
        name = model.get("name", f"Model_{i}")
        graph.nodes.append(ModelNode(
            name=name,
            index=i,
            variable_count=model.get("variable_count", len(model.get("variables", []))),
            connection_count=model.get("connection_count", len(model.get("connections", []))),
        ))

    # Detect shared variables (cross-model references)
    var_sets: List[Set[str]] = []
    for model in models:
        vars_ = set()
        for v in model.get("variables", []):
            vname = v.get("name", "") if isinstance(v, dict) else str(v)
            vars_.add(vname)
        var_sets.append(vars_)

    # Find overlapping variables between models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            shared = var_sets[i] & var_sets[j]
            if shared:
                source = graph.nodes[i].name
                target = graph.nodes[j].name
                graph.edges.append(ModelEdge(
                    source_model=source,
                    target_model=target,
                    shared_variables=sorted(shared),
                    label=f"shared: {', '.join(sorted(shared))}",
                ))
                logger.debug(f"Edge: {source} → {target} ({len(shared)} shared vars)")

    logger.info(f"📊 Dependency graph: {len(graph.nodes)} models, {len(graph.edges)} edges")
    return graph


def render_graph_from_file(
    file_path: str,
    output_format: str = "mermaid",
) -> str:
    """
    Parse a GNN file and render its model dependency graph.

    Args:
        file_path: Path to GNN file (possibly multi-model).
        output_format: "mermaid" or "text".

    Returns:
        Rendered graph string.
    """
    from pathlib import Path
    content = Path(file_path).read_text(encoding="utf-8")

    import sys
    src_dir = str(Path(__file__).parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from gnn.multimodel import parse_multimodel
    models = parse_multimodel(content, file_path=file_path)

    # Assign names from file sections or indices
    for i, m in enumerate(models):
        m.setdefault("name", f"Model_{i}")

    graph = build_dependency_graph(models, file_path=file_path)

    if output_format == "mermaid":
        return graph.to_mermaid()
    else:
        return graph.to_adjacency_list()
