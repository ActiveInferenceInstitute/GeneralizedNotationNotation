#!/usr/bin/env python3
"""Regenerate every manuscript figure deterministically.

Thin orchestrator that runs each ``scripts/manuscript_fig_*.py`` generator in turn,
so the manuscript's figures under ``output/figures/`` can be rebuilt from source with
a single command before rendering (the render pipeline does not regenerate figures).

It also writes ``output/figures/figure_registry.json``, the accessibility registry
the template's ``validate_figure_registry`` reads. The registry is *generated here*
rather than hand-authored precisely because it must not drift from the figures that
are actually built: the label, the produced filename and the generator that made it
all come from the same table below. Only ``alt_text`` is authored — it must describe
what a reader who cannot see the figure needs, in different words from the caption
(the caption is a title; the alt text says what the image shows).

Usage:
    python scripts/manuscript_build_figures.py
Exit code is non-zero if any generator fails, any expected PNG is missing, or the
registry does not cover every ``{#fig:...}`` label the manuscript declares.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIG_DIR = _PROJECT_ROOT / "output" / "figures"
_MANUSCRIPT_DIR = _PROJECT_ROOT / "manuscript"
_REGISTRY_PATH = _FIG_DIR / "figure_registry.json"

# (crossref label, generator script, expected PNG, alt text).
#
# The label is what the manuscript writes as {#fig:...}; a mismatch fails the
# build below rather than shipping an unregistered figure. The alt text is the
# only authored field, and is deliberately not the caption: a screen-reader user
# needs the figure's content, not its title.
_FIGURES = [
    (
        "fig:pipeline",
        "manuscript_fig_pipeline_dag.py",
        "gnn_pipeline_dag.png",
        "Directed acyclic graph of the numbered GNN pipeline steps. Nodes are the "
        "step modules in src/, arranged left to right from parsing and type "
        "checking, through validation, visualization, rendering and execution, to "
        "analysis, reporting and integration. Edges run from a step to every later "
        "step that consumes its artifacts, so each output can be traced back to the "
        "step that produced it.",
    ),
    (
        "fig:family_matrix",
        "manuscript_fig_family_framework.py",
        "gnn_family_framework_matrix.png",
        "Grid of model families (rows) against rendering backends (columns). A "
        "filled cell means the family declares that backend in "
        "input/model_family_manifest.json. The grid is sparse: most families "
        "declare a single backend, and only the continuous, hierarchical and "
        "gridworld families declare several. Cells record declared coverage, not "
        "profiled outcomes.",
    ),
    (
        "fig:backend_matrix",
        "manuscript_fig_backend_matrix.py",
        "gnn_backend_capability_matrix.png",
        "Table of the registered rendering backends, one row each, with columns "
        "for registry key, display name, implementation language, whether a "
        "render-output subdirectory exists under src/gnn/render/, and the backend's "
        "role in the cross-framework reference comparison. Highlighted rows are "
        "the backends the reliability gate profiles; a backend the family declares "
        "but the gate never profiles is marked separately.",
    ),
    (
        "fig:repo_metrics",
        "manuscript_fig_repo_metrics.py",
        "gnn_repo_metrics.png",
        "Horizontal bar chart on a logarithmic axis of repository-scale counts: "
        "pipeline steps, model families, registered backends, execution backends, "
        "MCP tools, source packages, test files, example models and documentation "
        "files. Each bar is annotated with its exact value; all values come from "
        "output/data/manuscript_variables.json.",
    ),
    (
        "fig:triple_play",
        "manuscript_fig_triple_play.py",
        "gnn_triple_play.png",
        "The Triple Play as one central GNN specification with three projections "
        "radiating from it: the readable text form, graphical visualizations of "
        "state-space and factor structure, and executable model code. All three "
        "point back to the single specification at the center.",
    ),
    (
        "fig:orchestration",
        "manuscript_fig_orchestration.py",
        "gnn_orchestration.png",
        "The three long-running orchestration contracts side by side — durable "
        "observation streams, resumable run sessions, and auditable container "
        "plans — each shown with the artifacts it generates and validates, and "
        "each stopping short of any live mutation.",
    ),
]

_FIG_LABEL_RE = re.compile(r"\{#(fig:[\w:-]+)")
# manuscript/SYNTAX.md is the authoring guide; its example embeds are not figures.
_LABEL_SCAN_SKIP = {"SYNTAX.md", "README.md", "AGENTS.md"}


def _declared_labels() -> set[str]:
    """Every ``{#fig:...}`` label the manuscript sections actually declare."""
    labels: set[str] = set()
    for md in sorted(_MANUSCRIPT_DIR.glob("*.md")):
        if md.name in _LABEL_SCAN_SKIP:
            continue
        labels.update(_FIG_LABEL_RE.findall(md.read_text(encoding="utf-8")))
    return labels


def _write_registry() -> list[str]:
    """Write figure_registry.json; return coverage problems (empty when clean)."""
    declared = _declared_labels()
    registered = {label for label, _, _, _ in _FIGURES}
    problems = [
        f"manuscript declares {label} with no registry entry"
        for label in sorted(declared - registered)
    ]
    problems += [
        f"registry entry {label} is declared by no manuscript section"
        for label in sorted(registered - declared)
    ]
    payload = {
        "schema_version": "gnn_figure_registry_v1",
        "figures": [
            {
                "label": label,
                "filename": png,
                "alt_text": alt,
                "generated_by": f"scripts/{script}",
            }
            for label, script, png, alt in _FIGURES
        ],
    }
    _REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return problems


def main() -> int:
    # Repo metrics reads output/data/manuscript_variables.json — make sure it exists.
    variables_json = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    if not variables_json.is_file():
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "z_generate_manuscript_variables.py"),
            ],
            cwd=str(_PROJECT_ROOT),
            check=False,
        )

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for _label, script, png, _alt in _FIGURES:
        script_path = _PROJECT_ROOT / "scripts" / script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(_PROJECT_ROOT),
            env={"MPLBACKEND": "Agg", **_environ()},
            check=False,
        )
        out = _FIG_DIR / png
        if result.returncode != 0:
            failures.append(f"{script} exited {result.returncode}")
        elif not (out.is_file() and out.stat().st_size > 5_000):
            failures.append(f"{script} did not produce {png} (>5KB)")
        else:
            print(f"  ✓ {png} ({out.stat().st_size // 1024} KB)")

    failures.extend(_write_registry())

    if failures:
        print("\n❌ Figure build FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print(f"\n✅ Built {len(_FIGURES)} manuscript figures in {_FIG_DIR}")
    print(f"✅ Wrote {_REGISTRY_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


def _environ() -> dict[str, str]:
    import os

    return os.environ.copy()


if __name__ == "__main__":
    raise SystemExit(main())
