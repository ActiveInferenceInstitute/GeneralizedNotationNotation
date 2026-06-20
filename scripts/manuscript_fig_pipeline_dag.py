#!/usr/bin/env python3
"""Deterministic generator for the GNN 25-step pipeline DAG figure.

Thin orchestrator: reads the real step roster, phases, and hard data
dependencies from ``src/STEP_INDEX.md`` (the master step table plus its
"Data Dependency Graph" mermaid block) and renders a layered, left-to-right
directed graph. No counts, names, or edges are hard-coded — everything is
parsed from the source-of-truth file.

Readability design: each step is a wide rounded-rectangle box (not a circle),
sized so the full ``N module_name`` label fits without truncation. Nodes are
placed on a hand-computed topological-depth layout (x = longest-path layer,
y = spread within the layer) so the flow reads left-to-right with no overlap.

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
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_INDEX = REPO_ROOT / "src" / "STEP_INDEX.md"
OUT_PNG = REPO_ROOT / "output" / "figures" / "gnn_pipeline_dag.png"

# Phase -> color (stable, ordered for the legend).
PHASE_COLORS: dict[str, str] = {
    "Global": "#9CA3AF",  # gray
    "Core": "#2563EB",  # blue
    "Analysis": "#7C3AED",  # purple
    "Simulation": "#DC2626",  # red
    "Output": "#059669",  # green
}

# Box geometry in data coordinates (half-width / half-height).
BOX_HALF_W = 1.25
BOX_HALF_H = 0.42

# Characters that fit on one line at the base font; longer labels shrink so the
# full module name always stays inside its box (e.g. ``advanced_visualization``).
LABEL_FIT_CHARS = 18
BASE_FONT = 8.0


def parse_steps(text: str) -> dict[int, dict[str, str]]:
    """Parse the master step table: step number -> {name, phase}."""
    steps: dict[int, dict[str, str]] = {}
    # Table rows look like: | 0 | [`0_template.py`](...) | [`template/`](...) | Global | ...
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


def compute_layout(
    g: nx.DiGraph, steps: dict[int, dict[str, str]]
) -> dict[int, tuple[float, float]]:
    """Layered left-to-right positions: x = topological depth, y = spread in layer."""
    depth: dict[int, int] = {}
    order = (
        list(nx.topological_sort(g))
        if nx.is_directed_acyclic_graph(g)
        else sorted(steps)
    )
    for n in order:
        preds = list(g.predecessors(n))
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1

    # Group nodes by layer, order within a layer by step number for determinism.
    layers: dict[int, list[int]] = {}
    for n in sorted(steps):
        layers.setdefault(depth[n], []).append(n)

    pos: dict[int, tuple[float, float]] = {}
    x_gap, y_gap = 4.0, 1.25
    for layer, nodes in layers.items():
        ordered = sorted(nodes)
        offset = (len(ordered) - 1) / 2.0
        for i, n in enumerate(ordered):
            pos[n] = (layer * x_gap, (offset - i) * y_gap)
    return pos


def draw_box(ax: Axes, x: float, y: float, label: str, color: str) -> None:
    """Draw one rounded-rectangle step node with a centered, full-width label."""
    box = FancyBboxPatch(
        (x - BOX_HALF_W, y - BOX_HALF_H),
        2 * BOX_HALF_W,
        2 * BOX_HALF_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=color,
        edgecolor="#1E293B",
        linewidth=1.1,
        zorder=3,
    )
    ax.add_patch(box)
    # Shrink the font for over-long labels so the full name stays inside the box.
    font = BASE_FONT
    if len(label) > LABEL_FIT_CHARS:
        font = BASE_FONT * LABEL_FIT_CHARS / len(label)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=font,
        fontweight="bold",
        color="white",
        zorder=4,
    )


def main() -> None:
    text = STEP_INDEX.read_text(encoding="utf-8")
    steps = parse_steps(text)
    if not steps:
        raise SystemExit("No steps parsed from STEP_INDEX.md")
    edges = parse_dependencies(text, set(steps))

    g: nx.DiGraph = nx.DiGraph()
    for s, meta in steps.items():
        g.add_node(s, **meta)
    g.add_edges_from(edges)

    pos = compute_layout(g, steps)

    fig, ax = plt.subplots(figsize=(22, 12))

    # --- edges first (under the boxes) ------------------------------------
    for a, b in edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        # Anchor on the right edge of source and left edge of target so the
        # arrows read as left-to-right flow and never cross box interiors.
        start = (xa + BOX_HALF_W, ya)
        end = (xb - BOX_HALF_W, yb)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle="arc3,rad=0.06",
            color="#94A3B8",
            linewidth=1.1,
            zorder=1,
        )
        ax.add_patch(arrow)

    # --- boxes ------------------------------------------------------------
    for n in sorted(steps):
        x, y = pos[n]
        label = f"{n}  {steps[n]['name']}"
        draw_box(ax, x, y, label, PHASE_COLORS[steps[n]["phase"]])

    # --- frame, legend, title ---------------------------------------------
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - BOX_HALF_W - 0.6, max(xs) + BOX_HALF_W + 0.6)
    ax.set_ylim(min(ys) - BOX_HALF_H - 0.8, max(ys) + BOX_HALF_H + 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    legend_handles = [mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    ax.legend(
        handles=legend_handles,
        title="Phase",
        loc="upper right",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )

    n_steps = len(steps)
    ax.set_title(
        "GNN 25-Step Processing Pipeline",
        fontsize=22,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        -0.02,
        f"{n_steps} steps ({min(steps)}–{max(steps)}) · "
        f"{len(edges)} hard data dependencies · "
        "left-to-right by topological layer, colored by execution phase",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#475569",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
