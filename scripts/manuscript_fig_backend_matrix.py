#!/usr/bin/env python3
"""Deterministic generator: GNN rendering backend registry table figure.

Thin orchestrator. Reads real data from:
  - src/gnn/render/framework_registry.py (FRAMEWORK_REGISTRY: registry key, display name)
  - src/gnn/render/<key>/ subdirectory existence (render-output presence)
  - input/model_family_manifest.json + src/gnn/pipeline/cross_framework_reliability.py
    (the single cross-framework comparison family selected by
    gnn.manuscript_variables.select_cross_framework_family, intersected with the
    reliability gate's MAINTAINED_FRAMEWORKS)

Writes output/figures/gnn_backend_capability_matrix.png at >=150 DPI.
No timestamps, no randomness; output is a pure function of the source files.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC = PROJECT_ROOT / "src"
RENDER_DIR = SRC / "render"
MANIFEST = PROJECT_ROOT / "input" / "model_family_manifest.json"
OUT_PNG = PROJECT_ROOT / "output" / "figures" / "gnn_backend_capability_matrix.png"


def load_registry() -> dict:
    """Import FRAMEWORK_REGISTRY from the real source module."""
    sys.path.insert(0, str(SRC))
    from gnn.render.framework_registry import FRAMEWORK_REGISTRY  # type: ignore

    return dict(FRAMEWORK_REGISTRY)


def cross_framework_backends() -> tuple[str, list[str], list[str]]:
    """Return ``(family_name, profiled_keys, declared_only_keys)``.

    The highlighted set is the ONE reference-comparison family — selected by the
    same helper the token map uses, so the figure and the prose cannot name
    different sets — intersected with the reliability gate's
    ``MAINTAINED_FRAMEWORKS``. Previously this unioned every multi-framework
    family and the footer called the union "the cross-framework comparison
    family", producing a third cardinality for one concept inside one document.
    """
    from gnn.manuscript_variables import select_cross_framework_family

    manifest = json.loads(MANIFEST.read_text())
    family = select_cross_framework_family(manifest.get("families", []))
    if family is None:
        return "", [], []
    declared = [
        k.strip() for k in str(family.get("frameworks", "")).split(",") if k.strip()
    ]

    gate = (SRC / "pipeline" / "cross_framework_reliability.py").read_text()
    block = re.search(r"MAINTAINED_FRAMEWORKS = \(([^)]*)\)", gate)
    maintained = set(re.findall(r'"([a-z_]+)"', block.group(1))) if block else set()
    if not maintained:
        return str(family.get("name", "")), declared, []
    profiled = [k for k in declared if k in maintained]
    declared_only = [k for k in declared if k not in maintained]
    return str(family.get("name", "")), profiled, declared_only


def main() -> None:
    registry = load_registry()
    family_name, profiled, declared_only = cross_framework_backends()
    profiled_set = set(profiled)
    declared_only_set = set(declared_only)

    rows = []
    for key, spec in registry.items():
        subdir = RENDER_DIR / key
        has_dir = subdir.is_dir()
        if key in profiled_set:
            cross_cell = "profiled"
        elif key in declared_only_set:
            cross_cell = "declared"
        else:
            cross_cell = ""
        rows.append(
            [
                key,
                spec["name"],
                spec.get("language", ""),
                "yes" if has_dir else "no",
                cross_cell,
            ]
        )

    headers = [
        "Registry Key",
        "Display Name",
        "Language",
        "Render Subdir",
        "Cross-Framework",
    ]
    subtitle = f"cross-framework reference family: {family_name}" if family_name else ""

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    n_rows = len(rows)
    fig_h = 1.4 + 0.42 * n_rows
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    ax.set_title(
        "GNN Rendering Backend Registry" + (f"\n{subtitle}" if subtitle else ""),
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.20, 0.22, 0.14, 0.18, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    header_bg = "#1f3b57"
    cross_bg = "#fde9a8"
    declared_bg = "#e8e2cd"
    nosub_bg = "#f6d4d4"
    stripe_bg = "#eef2f6"

    for (r, c), cell in (
        table.get_cells().items()
        if hasattr(table, "get_cells")
        else table.get_celld().items()
    ):
        cell.set_edgecolor("#9aa6b2")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        row = rows[r - 1]
        is_cross = row[4] == "profiled"
        is_declared_only = row[4] == "declared"
        no_sub = row[3] == "no"
        if is_cross:
            cell.set_facecolor(cross_bg)
        elif is_declared_only:
            cell.set_facecolor(declared_bg)
        elif no_sub and c == 3:
            cell.set_facecolor(nosub_bg)
        elif r % 2 == 0:
            cell.set_facecolor(stripe_bg)
        else:
            cell.set_facecolor("white")
        if c == 0:
            cell.set_text_props(fontfamily="monospace")

    n_backends = len(rows)
    n_with_dir = sum(1 for row in rows if row[3] == "yes")
    n_cross = len(profiled)
    caption = (
        f"{n_backends} registered backends · {n_with_dir} with a render-output "
        f"subdir under src/gnn/render/ · highlighted {n_cross} backends are profiled "
        f"in the {family_name or 'cross-framework'} reference comparison"
    )
    if declared_only:
        # Second line: bbox_inches="tight" grows the whole figure to fit a long
        # single-line caption, which shrinks the table when the PNG is scaled
        # into a PDF column.
        caption += (
            f"\n{', '.join(declared_only)} declared by the family but outside "
            "MAINTAINED_FRAMEWORKS, so never profiled"
        )
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
        linespacing=1.5,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
