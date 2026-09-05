"""Pure connection-statistics for GNN visualization (no plotting dependencies).

The degree-based summary computed here is the one reported by the
package-root :func:`visualization._generate_network_statistics` alias and
pinned by ``tests/visualization/test_visualization_module_info.py``. It is
intentionally distinct from the richer ``graph.network_visualizations``
metrics helper, which adds networkx topology counts.
"""

from __future__ import annotations

from typing import Any, Dict


def compute_connection_statistics(
    variables: Dict[str, Any], connections: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute degree-based statistics from variables and connections.

    Parameters:
        variables: Mapping of variable name to variable info. Only the keys
            are used (for the ``total_nodes`` / ``isolated_nodes`` counts).
        connections: Connection dicts; each must carry ``source`` and
            ``target`` keys (missing keys fall back to ``"unknown"``).

    Returns:
        Summary dict with ``total_nodes``, ``total_connections``,
        ``average_degree``, ``max_degree``, ``min_degree``,
        ``node_degree_distribution``, ``isolated_nodes`` and ``hub_nodes``.
    """
    node_degrees: dict[Any, int] = {}
    for conn in connections:
        source = conn.get("source", "unknown")
        target = conn.get("target", "unknown")
        node_degrees[source] = node_degrees.get(source, 0) + 1
        node_degrees[target] = node_degrees.get(target, 0) + 1

    if node_degrees:
        degrees = list(node_degrees.values())
        return {
            "total_nodes": len(variables),
            "total_connections": len(connections),
            "average_degree": sum(degrees) / len(degrees),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "node_degree_distribution": dict(node_degrees),
            "isolated_nodes": len(
                [v for v in variables.keys() if v not in node_degrees]
            ),
            "hub_nodes": [node for node, degree in node_degrees.items() if degree > 2],
        }

    return {
        "total_nodes": len(variables),
        "total_connections": len(connections),
        "average_degree": 0,
        "max_degree": 0,
        "min_degree": 0,
        "node_degree_distribution": {},
        "isolated_nodes": len(variables),
        "hub_nodes": [],
    }
