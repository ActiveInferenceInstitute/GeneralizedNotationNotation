#!/usr/bin/env python3
"""Deterministic generator for the model-family x rendering-backend coverage matrix.

Thin orchestrator: reads real data from
  - input/model_family_manifest.json  (rows = families, "frameworks" field = declared backends)
  - src/render/framework_registry.py  (cols = canonical backend keys)
and renders an annotated coverage heatmap. A cell is "covered" when a family
declares that framework in its comma-separated "frameworks" field.

Headless (MPLBACKEND=Agg), deterministic (no timestamps/randomness).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "input" / "model_family_manifest.json"
REGISTRY_PATH = REPO_ROOT / "src" / "render" / "framework_registry.py"
OUTPUT_PATH = REPO_ROOT / "output" / "figures" / "gnn_family_framework_matrix.png"


def _load_registry_keys() -> list[str]:
    """Load canonical backend keys in registry order from framework_registry.py."""
    spec = importlib.util.spec_from_file_location(
        "gnn_framework_registry", REGISTRY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.get_supported_frameworks())


def _load_families() -> list[dict]:
    """Load family records from the manifest, preserving declared order."""
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["families"])


def main() -> Path:
    backends = _load_registry_keys()
    families = _load_families()

    backend_names: dict[str, str] = {}
    spec = importlib.util.spec_from_file_location(
        "gnn_framework_registry", REGISTRY_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in backends:
        backend_names[key] = module.FRAMEWORK_REGISTRY[key]["name"]

    family_labels = [f["name"] for f in families]
    declared = [
        {tok.strip() for tok in str(f["frameworks"]).split(",") if tok.strip()}
        for f in families
    ]

    n_rows = len(family_labels)
    n_cols = len(backends)
    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for r, decl in enumerate(declared):
        for c, key in enumerate(backends):
            if key in decl:
                matrix[r, c] = 1.0

    fig_w = max(8.0, 1.05 * n_cols + 2.5)
    fig_h = max(5.0, 0.62 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.colors.ListedColormap(["#f0f0f0", "#2a7a4f"])
    ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        [backend_names[k] for k in backends], rotation=45, ha="right", fontsize=10
    )
    ax.set_yticklabels(family_labels, fontsize=10)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for r in range(n_rows):
        for c in range(n_cols):
            if matrix[r, c] > 0.5:
                ax.text(
                    c,
                    r,
                    "✓",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                )

    covered = int(matrix.sum())
    total = n_rows * n_cols
    ax.set_xlabel("Rendering Backend", fontsize=11)
    ax.set_ylabel("Model Family", fontsize=11)
    ax.set_title(
        "Model Family × Rendering Backend Coverage", fontsize=13, fontweight="bold"
    )
    fig.text(
        0.5,
        0.005,
        f"Declared coverage cells: {covered} of {total} "
        f"({n_rows} families × {n_cols} backends)",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(str(main()))
