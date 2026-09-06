"""Shared model-content and graph utilities for the validation module.

Hosts the helpers more than one validator needs: canonical GNN content
extraction from parsed model dictionaries, bounded score clamping, friendly
file naming, and exact directed-cycle detection (Tarjan strongly connected
components). Keeping them here means every validator interprets parsed
model dictionaries identically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DirectedEdge",
    "clamp01",
    "cycle_nodes",
    "display_file_name",
    "extract_content_from_dict",
]


def clamp01(value: float) -> float:
    """Clamp a heuristic score into the inclusive [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def display_file_name(file_path: str) -> str:
    """Return the basename for a path, preserving the ``unknown`` sentinel."""
    return Path(file_path).name if file_path != "unknown" else "unknown"


def extract_content_from_dict(model_data: Mapping[str, Any]) -> str:
    """Extract GNN text content from a parsed model dictionary.

    Prefers reconstructing the original document from ``raw_sections``; when
    only structured ``variables``/``connections`` are available, renders a
    best-effort canonical approximation so downstream regex-based validators
    still have text to inspect. Returns an empty string when the dictionary
    carries neither representation.
    """
    raw_sections = model_data.get("raw_sections", {})
    if raw_sections:
        return "\n\n".join(f"## {name}\n{body}" for name, body in raw_sections.items())

    # Recovery: try to get variables and connections
    variables = model_data.get("variables", [])
    connections = model_data.get("connections", [])

    if variables or connections:
        content_parts = []

        # Add variables
        if variables:
            var_lines: list[Any] = []
            for var in variables:
                name = var.get("name", "Unknown")
                var_type = var.get("var_type", "unknown")
                dimensions = var.get("dimensions", [])
                dim_str = (
                    "[" + ", ".join(map(str, dimensions)) + "]" if dimensions else ""
                )
                var_lines.append(f"{name}{dim_str} # {var_type}")
            content_parts.append("StateSpaceBlock:\n" + "\n".join(var_lines))

        # Add connections
        if connections:
            conn_lines: list[Any] = []
            for conn in connections:
                sources = conn.get("source_variables") or ["?"]
                targets = conn.get("target_variables") or ["?"]
                source = (
                    sources[0] if len(sources) == 1 else "(" + ",".join(sources) + ")"
                )
                target = (
                    targets[0] if len(targets) == 1 else "(" + ",".join(targets) + ")"
                )
                operator = "-" if conn.get("connection_type") == "undirected" else ">"
                annotation = conn.get("annotation")
                suffix = f":{annotation}" if annotation else ""
                conn_lines.append(f"{source} {operator} {target}{suffix}")
            content_parts.append("Connections:\n" + "\n".join(conn_lines))

        return "\n\n".join(content_parts)

    # Final recovery: return empty string
    return ""


@dataclass(frozen=True)
class DirectedEdge:
    """One directed edge of a model dependency graph."""

    source: str
    target: str


def cycle_nodes(nodes: Sequence[str], edges: Iterable[DirectedEdge]) -> list[str]:
    """Return exactly the nodes that participate in directed cycles.

    Uses Tarjan's strongly connected components algorithm: a node is cyclic
    when it belongs to a component of size greater than one or has a
    self-edge. The result preserves the first-seen order of ``nodes`` (plus
    any edge-only endpoints, in first-seen order), so callers can report
    cycles deterministically. Nodes that merely lead into a cycle are
    excluded.
    """
    graph: dict[str, list[str]] = {}
    for name in nodes:
        graph.setdefault(name, [])
    for edge in edges:
        graph.setdefault(edge.source, []).append(edge.target)
        graph.setdefault(edge.target, [])

    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cyclic: set[str] = set()

    def strong_connect(node: str) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in indices:
                strong_connect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in graph.get(node, []):
            cyclic.update(component)

    for node in graph:
        if node not in indices:
            strong_connect(node)
    return [node for node in graph if node in cyclic]
