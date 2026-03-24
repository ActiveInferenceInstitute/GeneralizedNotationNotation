"""Bipartite sketch: GNN variables vs named parameter tensors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except (ImportError, RecursionError, AttributeError, ValueError):
    nx = None
    NETWORKX_AVAILABLE = False

from visualization.plotting.utils import safe_tight_layout, save_plot_safely

logger = logging.getLogger(__name__)


def generate_variable_parameter_bipartite(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """
    Draw variables (left) and parameter matrices (right).
    Edges: parameter name links to a variable with the same symbol when present.
    """
    out: List[str] = []
    if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE or plt is None or nx is None:
        return out

    variables = parsed_data.get("variables") or []
    parameters = parsed_data.get("parameters") or []
    if not parameters:
        return out

    var_names = {
        v.get("name")
        for v in variables
        if isinstance(v, dict) and v.get("name")
    }
    param_names_set = {
        str(p["name"])
        for p in parameters
        if isinstance(p, dict) and p.get("name")
    }
    if not param_names_set:
        return out

    B = nx.Graph()
    left_nodes: List[str] = []
    for v in sorted(var_names):
        B.add_node(v, kind="variable")
        left_nodes.append(v)
    right_nodes: List[str] = []
    for p in sorted(param_names_set):
        pid = f"θ:{p}"
        B.add_node(pid, kind="parameter", label=p)
        right_nodes.append(pid)
        if p in var_names:
            B.add_edge(p, pid)

    if B.number_of_nodes() == 0:
        return out

    try:
        plt.figure(figsize=(12, 8))
        try:
            from networkx.algorithms import bipartite as nx_bipartite

            top = set(left_nodes) if left_nodes else set(right_nodes)
            pos = nx_bipartite.bipartite_layout(B, top, align="vertical", scale=2.0)
        except Exception:
            pos = nx.spring_layout(B, seed=42)

        nx.draw_networkx_nodes(
            B,
            pos,
            nodelist=left_nodes,
            node_color="lightblue",
            node_size=900,
            alpha=0.9,
        )
        nx.draw_networkx_nodes(
            B,
            pos,
            nodelist=right_nodes,
            node_color="wheat",
            node_size=900,
            alpha=0.9,
        )
        labels = {}
        for n in B.nodes():
            d = B.nodes[n]
            if d.get("kind") == "parameter":
                labels[n] = d.get("label", n)
            else:
                labels[n] = str(n)
        nx.draw_networkx_labels(B, pos, labels=labels, font_size=9)
        nx.draw_networkx_edges(B, pos, alpha=0.6, width=1.5)
        plt.title(f"{model_name} — variables ↔ parameters (name match)")
        plt.axis("off")
        safe_tight_layout()
        path = output_dir / f"{model_name}_variable_parameter_bipartite.png"
        save_plot_safely(path, dpi=300, bbox_inches="tight")
        plt.close()
        out.append(str(path))
    except Exception as e:
        logger.debug("Bipartite plot skipped: %s", e)
        try:
            plt.close()
        except Exception:
            pass
    return out
