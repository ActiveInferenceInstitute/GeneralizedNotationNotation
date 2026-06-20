#!/usr/bin/env python3
"""Deterministic generator: GNN rendering backend registry table figure.

Thin orchestrator. Reads real data from:
  - src/render/framework_registry.py (FRAMEWORK_REGISTRY: registry key, display name)
  - src/render/<key>/ subdirectory existence (render-output presence)
  - input/model_family_manifest.json (cross-framework comparison family:
    the family whose comma-separated "frameworks" lists more than one backend)

Writes output/figures/gnn_backend_capability_matrix.png at >=150 DPI.
No timestamps, no randomness; output is a pure function of the source files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
RENDER_DIR = SRC / "render"
MANIFEST = PROJECT_ROOT / "input" / "model_family_manifest.json"
OUT_PNG = PROJECT_ROOT / "output" / "figures" / "gnn_backend_capability_matrix.png"


def load_registry() -> dict:
    """Import FRAMEWORK_REGISTRY from the real source module."""
    sys.path.insert(0, str(SRC))
    from render.framework_registry import FRAMEWORK_REGISTRY  # type: ignore

    return dict(FRAMEWORK_REGISTRY)


def cross_framework_backends() -> set[str]:
    """Backends in the manifest family whose 'frameworks' lists >1 backend."""
    manifest = json.loads(MANIFEST.read_text())
    cross: set[str] = set()
    for fam in manifest.get("families", []):
        raw = fam.get("frameworks", "")
        parts = (
            [p.strip() for p in raw.split(",") if p.strip()]
            if isinstance(raw, str)
            else list(raw)
        )
        if len(parts) > 1:
            cross.update(parts)
    return cross


def main() -> None:
    registry = load_registry()
    cross = cross_framework_backends()

    rows = []
    for key, spec in registry.items():
        subdir = RENDER_DIR / key
        has_dir = subdir.is_dir()
        rows.append(
            [
                key,
                spec["name"],
                spec.get("language", ""),
                "yes" if has_dir else "no",
                "yes" if key in cross else "",
            ]
        )

    headers = [
        "Registry Key",
        "Display Name",
        "Language",
        "Render Subdir",
        "Cross-Framework",
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    n_rows = len(rows)
    fig_h = 1.4 + 0.42 * n_rows
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    ax.set_title(
        "GNN Rendering Backend Registry",
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
    nosub_bg = "#f6d4d4"
    stripe_bg = "#eef2f6"

    for (r, c), cell in table.get_cells().items() if hasattr(table, "get_cells") else table.get_celld().items():
        cell.set_edgecolor("#9aa6b2")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        row = rows[r - 1]
        is_cross = row[4] == "yes"
        no_sub = row[3] == "no"
        if is_cross:
            cell.set_facecolor(cross_bg)
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
    n_cross = sum(1 for row in rows if row[4] == "yes")
    caption = (
        f"{n_backends} registered backends · {n_with_dir} with a render-output "
        f"subdir under src/render/ · highlighted {n_cross} backends form the "
        f"cross-framework comparison family"
    )
    fig.text(0.5, 0.03, caption, ha="center", fontsize=8.5, color="#444444")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(str(OUT_PNG))


if __name__ == "__main__":
    main()
