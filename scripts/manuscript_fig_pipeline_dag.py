#!/usr/bin/env python3
"""Deterministic generator for the GNN 25-step pipeline DAG figure.

Thin orchestrator: reads the real step roster, phases, and hard data
dependencies from ``src/STEP_INDEX.md`` (the master step table plus its
"Data Dependency Graph" mermaid block) and renders a layered, left-to-right
directed graph. No counts, names, or edges are hard-coded — everything is
parsed from the source-of-truth file.

Output: output/figures/gnn_pipeline_dag.png (>=150 DPI, headless).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_INDEX = REPO_ROOT / "src" / "STEP_INDEX.md"
OUT_PNG = REPO_ROOT / "output" / "figures" / "gnn_pipeline_dag.png"

# Phase -> color (stable, ordered for the legend).
PHASE_COLORS: dict[str, str] = {
    "Global": "#9CA3AF",       # gray
    "Core": "#2563EB",         # blue
    "Analysis": "#7C3AED",     # purple
    "Simulation": "#DC2626",   # red
    "Output": "#059669",       # green
}


def parse_steps(text: str) -> dict[int, dict[str, str]]:
    """Parse the master step table: step number -> {name, phase}."""
    steps: dict[int, dict[str, str]] = {}
    # Table rows look like: | 0 | [`0_template.py`](...) | [`template/`](...) | Global | Pipeline template ...
    row_re = re.compile(r"^\|\s*(\d+)\s*\|(.+)$")
    for line in text.splitlines():
        m = row_re.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        cells = [c.strip() for c in m.group(2).split("|")]
        # cells: [Script, Module Dir, Phase, Purpose, ...]
        if len(cells) < 3:
            continue
        phase = cells[2]
        if phase not in PHASE_COLORS:
            continue  # skip legend/other tables that happen to start with a digit
        # Short label from the module dir cell, e.g. [`template/`](...) -> template
        mod = cells[1]
        mm = re.search(r"`([^`/]+)/?`", mod)
        name = mm.group(1).rstrip("/") if mm else f"step{step}"
        steps[step] = {"name": name, "phase": phase}
    return steps


def parse_dependencies(text: str, valid: set[int]) -> list[tuple[int, int]]:
    """Parse hard prerequisite edges from the Data Dependency Graph mermaid block."""
    # Isolate the dependency-graph mermaid block (graph TD ... ).
    start = text.find("## Data Dependency Graph")
    block = text[start:] if start != -1 else text
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # Edges like: S3[3 GNN Parse] --> S5[5 Type Check]  or  S3 --> S4[4 Registry]
    # Node labels (the [...] block) are optional on either endpoint, and an
    # edge label |"optional enrichment"| may sit between the arrow and target.
    edge_re = re.compile(
        r"S(\d+)(?:\[[^\]]*\])?\s*-->(?:\s*\|[^|]*\|)?\s*S(\d+)(?:\[[^\]]*\])?"
    )
    for src, dst in edge_re.findall(block):
        a, b = int(src), int(dst)
        if a in valid and b in valid and (a, b) not in seen:
            edges.append((a, b))
            seen.add((a, b))
    return edges


def main() -> None:
    text = STEP_INDEX.read_text(encoding="utf-8")
    steps = parse_steps(text)
    if not steps:
        raise SystemExit("No steps parsed from STEP_INDEX.md")
    edges = parse_dependencies(text, set(steps))

    g = nx.DiGraph()
    for s, meta in steps.items():
        g.add_node(s, **meta)
    g.add_edges_from(edges)

    # --- deterministic layered layout -------------------------------------
    # x = topological "depth" (longest path from a source); y = spread within layer.
    # Add a virtual chain on step number so isolated/sequence ordering is stable.
    depth: dict[int, int] = {}
    for n in nx.topological_sort(g) if nx.is_directed_acyclic_graph(g) else sorted(steps):
        preds = list(g.predecessors(n))
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1

    # Group nodes by layer, order within a layer by step number for determinism.
    layers: dict[int, list[int]] = {}
    for n in sorted(steps):
        layers.setdefault(depth[n], []).append(n)

    pos: dict[int, tuple[float, float]] = {}
    x_gap, y_gap = 2.4, 1.6
    for layer, nodes in layers.items():
        nodes = sorted(nodes)
        offset = (len(nodes) - 1) / 2.0
        for i, n in enumerate(nodes):
            pos[n] = (layer * x_gap, (offset - i) * y_gap)

    node_colors = [PHASE_COLORS[steps[n]["phase"]] for n in g.nodes()]
    labels = {n: f"{n}\n{steps[n]['name']}" for n in g.nodes()}

    fig, ax = plt.subplots(figsize=(18, 10))
    nx.draw_networkx_edges(
        g, pos, ax=ax, edge_color="#94A3B8", width=1.3,
        arrows=True, arrowstyle="-|>", arrowsize=12,
        connectionstyle="arc3,rad=0.08", min_source_margin=16, min_target_margin=16,
    )
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=2300,
        edgecolors="#1E293B", linewidths=1.2,
    )
    nx.draw_networkx_labels(
        g, pos, labels=labels, ax=ax, font_size=7.5,
        font_color="white", font_weight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles, title="Phase", loc="upper left",
        bbox_to_anchor=(1.005, 1.0), frameon=True, fontsize=10, title_fontsize=11,
    )

    n_steps = len(steps)
    ax.set_title(
        "GNN 25-Step Processing Pipeline",
        fontsize=20, fontweight="bold", pad=16,
    )
    ax.text(
        0.5, -0.04,
        f"{n_steps} steps ({min(steps)}–{max(steps)}) · "
        f"{len(edges)} hard data dependencies · colored by execution phase",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#475569",
    )
    ax.set_axis_off()
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
