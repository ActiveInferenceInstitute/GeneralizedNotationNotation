#!/usr/bin/env python3
"""Deterministic thin-orchestrator generator for the v3.0.0 orchestration figure.

Renders the v3.0.0 ("Long-Running Orchestration") architecture: three
safe-by-design ``src/pipeline/`` contracts —

1. Durable Observation Streams (StreamManifest + ExecutionTrace + replay),
2. Resumable Run Sessions (checkpoint / resume / status / safe cleanup),
3. Auditable Container Plans (generate -> static security review -> rollback) —

all sitting inside a labeled boundary that emphasizes the core invariant:
data only, with no live infrastructure mutation.

Headless (MPLBACKEND=Agg), deterministic (fixed positions, no timestamps or
random state), saves a >=150 DPI PNG to ``output/figures/gnn_orchestration.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "output" / "figures" / "gnn_orchestration.png"


def draw_box(
    ax: Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 11,
    fontweight: str = "bold",
    textcolor: str = "white",
) -> None:
    """Draw a rounded box centered at ``xy`` with wrapped text."""
    x, y = xy
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=4,
        wrap=True,
    )


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
) -> None:
    """Draw a dark directional arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color=color,
        shrinkA=4,
        shrinkB=4,
        zorder=2,
    )
    ax.add_patch(arrow)


def main() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Safe-by-design boundary: data only, no live infrastructure mutation.
    boundary = FancyBboxPatch(
        (0.35, 0.85),
        11.3,
        5.35,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#f1f5f9",
        edgecolor="#0f766e",
        linewidth=2.4,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(boundary)
    ax.text(
        6.0,
        5.9,
        "Safe-by-design boundary — data only, no live infrastructure mutation",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0f766e",
        zorder=2,
    )

    # Each contract: x position, header title, color, ordered stage chain.
    contracts: list[tuple[float, str, str, list[str]]] = [
        (
            2.2,
            "1. Durable\nObservation Streams\n(src/pipeline/\ndurable_streams.py)",
            "#1e3a8a",
            ["StreamManifest", "ExecutionTrace", "replay"],
        ),
        (
            6.0,
            "2. Resumable\nRun Sessions\n(src/pipeline/\nrun_session.py)",
            "#7c3aed",
            ["checkpoint", "resume", "status", "safe cleanup"],
        ),
        (
            9.8,
            "3. Auditable\nContainer Plans\n(src/pipeline/\ncontainer_plan.py)",
            "#b45309",
            ["generate", "static security review", "rollback"],
        ),
    ]

    header_y = 4.55
    stage_top = 3.65
    stage_h = 0.62
    stage_gap = 0.18

    for x, title, color, stages in contracts:
        draw_box(
            ax,
            (x, header_y),
            3.2,
            1.35,
            title,
            facecolor=color,
            edgecolor="#0f172a",
            fontsize=10,
        )
        draw_arrow(ax, (x, header_y - 0.7), (x, stage_top + stage_h / 2 + 0.08))

        y = stage_top
        for i, stage in enumerate(stages):
            draw_box(
                ax,
                (x, y),
                3.0,
                stage_h,
                stage,
                facecolor="#475569",
                edgecolor="#0f172a",
                fontsize=10,
                textcolor="white",
            )
            if i < len(stages) - 1:
                next_y = y - (stage_h + stage_gap)
                draw_arrow(
                    ax,
                    (x, y - stage_h / 2 - 0.02),
                    (x, next_y + stage_h / 2 + 0.02),
                )
            y -= stage_h + stage_gap

    ax.set_title(
        "GNN v3.0.0 — Long-Running Orchestration",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )
    ax.text(
        6.0,
        0.32,
        "Three safe-by-design pipeline contracts with additive live wiring, a strict "
        "acceptance gate, and 3 new MCP tools — all data-only.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#334155",
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUT_PATH


if __name__ == "__main__":
    out = main()
    print(str(out))
