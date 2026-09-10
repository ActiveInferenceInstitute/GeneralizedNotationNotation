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

Each entry also carries build provenance, which is what stops a stale figure from
shipping. ``png_sha256`` is the digest of the PNG this build produced, and
``consumed_tokens`` is the ``{key: value}`` map the generator actually read out of
``output/data/manuscript_variables.json`` (observed by
``scripts/lib/manuscript_figure_tokens.py``, not declared).
``src/tests/test_manuscript_figure_freshness.py`` re-checks both against the
committed PNG and the live token map, so a figure built before a count moved fails
the suite. Without it, ``fig:repo_metrics`` shipped "365 test files" on the same PDF
page as prose reading 367.

Usage:
    python scripts/manuscript_build_figures.py
Exit code is non-zero if any generator fails, any expected PNG is missing, or the
registry does not cover every ``{#fig:...}`` label the manuscript declares.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript import RepositorySnapshot  # noqa: E402
from scripts.lib.manuscript_exclusions import AUTHORING_GUIDE_SKIP  # noqa: E402

_FIG_DIR = _PROJECT_ROOT / "output" / "figures"
_MANUSCRIPT_DIR = _PROJECT_ROOT / "manuscript"
_REGISTRY_PATH = _FIG_DIR / "figure_registry.json"
_PROVENANCE_ENV = "GNN_FIGURE_TOKEN_PROVENANCE"

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
# Authoring guides (SYNTAX.md etc.) hold example embeds, not figures.
_LABEL_SCAN_SKIP = AUTHORING_GUIDE_SKIP


def _declared_labels() -> set[str]:
    """Every ``{#fig:...}`` label the manuscript sections actually declare."""
    labels: set[str] = set()
    for md in sorted(_MANUSCRIPT_DIR.glob("*.md")):
        if md.name in _LABEL_SCAN_SKIP:
            continue
        labels.update(_FIG_LABEL_RE.findall(md.read_text(encoding="utf-8")))
    return labels


def _sha256(path: Path) -> str:
    """Digest of a built artifact, so a swapped or stale PNG is detectable.

    Returns the empty string when the figure was not produced: this build has
    already recorded that as a failure and exits non-zero, and an empty digest
    can never match a real PNG in the freshness gate.
    """
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(provenance: dict[str, dict[str, str]]) -> list[str]:
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
        "schema_version": "gnn_figure_registry_v2",
        "figures": [
            {
                "label": label,
                "filename": png,
                "alt_text": alt,
                "generated_by": f"scripts/{script}",
                "png_sha256": _sha256(_FIG_DIR / png),
                "consumed_tokens": provenance.get(label, {}),
            }
            for label, script, png, alt in _FIGURES
        ],
    }
    _REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return problems


def main() -> int:
    # Regenerate the token map on EVERY build. The old "only when missing"
    # skip plus check=False left a stale output/data/manuscript_variables.json
    # feeding every generator silently. A fresh run also fails loudly below if
    # the map describes a different commit than the one being built from.
    variables_json = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    try:
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "z_generate_manuscript_variables.py"),
            ],
            cwd=str(_PROJECT_ROOT),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "❌ scripts/z_generate_manuscript_variables.py failed with exit code "
            f"{exc.returncode}; fix the producer (see its stderr above) before "
            "building figures against a token map"
        ) from exc
    try:
        generated = json.loads(variables_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"❌ {variables_json} is unreadable after regeneration ({exc}); "
            "figures cannot be built against an auditable token map"
        ) from exc
    head = RepositorySnapshot(_PROJECT_ROOT)
    recorded = str(generated.get("GNN_GIT_COMMIT", "unknown"))
    if recorded == "unknown" or recorded != head.commit:
        raise SystemExit(
            "❌ output/data/manuscript_variables.json was generated for commit "
            f"{recorded!r} but this build reads {head.commit!r} — the figures "
            "would print numbers the prose does not. Commit or clean the tree, "
            "then rerun: python -m scripts.manuscript_build_figures"
        )

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    provenance: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, script, png, _alt in _FIGURES:
            script_path = _PROJECT_ROOT / "scripts" / script
            # Each generator writes the token keys it actually read here.
            token_record = Path(tmp) / f"{label.replace(':', '_')}.json"
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(_PROJECT_ROOT),
                env={
                    "MPLBACKEND": "Agg",
                    _PROVENANCE_ENV: str(token_record),
                    **_environ(),
                },
                check=False,
            )
            out = _FIG_DIR / png
            if result.returncode != 0:
                failures.append(f"{script} exited {result.returncode}")
                continue
            if not (out.is_file() and out.stat().st_size > 5_000):
                failures.append(f"{script} did not produce {png} (>5KB)")
                continue
            # Absent record = the generator reads no tokens (e.g. fig:orchestration).
            provenance[label] = (
                json.loads(token_record.read_text(encoding="utf-8"))
                if token_record.is_file()
                else {}
            )
            print(f"  ✓ {png} ({out.stat().st_size // 1024} KB)")

    failures.extend(_write_registry(provenance))

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
