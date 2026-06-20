#!/usr/bin/env python3
"""Deterministic generator for the GNN repository-scale metrics figure.

Thin orchestrator: reads repository-scale counts STRICTLY from the deterministic
producer output (output/data/manuscript_variables.json) and renders a horizontal
bar chart. No counts are hard-coded; only the metric keys and human-readable
labels live here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo root = parent of this scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "output" / "data" / "manuscript_variables.json"
OUTPUT_PATH = REPO_ROOT / "output" / "figures" / "gnn_repo_metrics.png"

# (json_key, human-readable label) — order is rendering order (top to bottom).
METRICS: list[tuple[str, str]] = [
    ("GNN_STEP_COUNT", "Pipeline steps"),
    ("GNN_FAMILY_COUNT", "Model families"),
    ("GNN_BACKEND_COUNT", "Execution backends"),
    ("GNN_MCP_TOOL_COUNT", "MCP tools"),
    ("GNN_SRC_PACKAGE_COUNT", "Source packages"),
    ("GNN_TEST_FILE_COUNT", "Test files"),
    ("GNN_EXAMPLE_COUNT", "Examples"),
    ("GNN_DOC_FILE_COUNT", "Documentation files"),
]


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    labels = [label for _, label in METRICS]
    values = [int(data[key]) for key, _ in METRICS]
    version = str(data["GNN_VERSION"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Render top-to-bottom in declared order by reversing for the y-axis.
    y_positions = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(y_positions, values, color="#2c7fb8", edgecolor="#0f3d5c")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Count (log scale)", fontsize=11)
    ax.set_title(
        f"GeneralizedNotationNotation at a Glance (v{version})",
        fontsize=14,
        fontweight="bold",
    )

    # Log scale keeps small (steps=25) and large (docs=606) bars all legible.
    ax.set_xscale("log")
    ax.set_xlim(1, max(values) * 1.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="#0f3d5c",
        )

    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
