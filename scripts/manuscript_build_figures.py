#!/usr/bin/env python3
"""Regenerate every manuscript figure deterministically.

Thin orchestrator that runs each ``scripts/manuscript_fig_*.py`` generator in turn,
so the manuscript's figures under ``output/figures/`` can be rebuilt from source with
a single command before rendering (the render pipeline does not regenerate figures).

Usage:
    python scripts/manuscript_build_figures.py
Exit code is non-zero if any generator fails or any expected PNG is missing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIG_DIR = _PROJECT_ROOT / "output" / "figures"

# (generator script, expected PNG) — keep in sync with the manuscript figure embeds.
_FIGURES = [
    ("manuscript_fig_pipeline_dag.py", "gnn_pipeline_dag.png"),
    ("manuscript_fig_family_framework.py", "gnn_family_framework_matrix.png"),
    ("manuscript_fig_backend_matrix.py", "gnn_backend_capability_matrix.png"),
    ("manuscript_fig_repo_metrics.py", "gnn_repo_metrics.png"),
    ("manuscript_fig_triple_play.py", "gnn_triple_play.png"),
]


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
    for script, png in _FIGURES:
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

    if failures:
        print("\n❌ Figure build FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print(f"\n✅ Built {len(_FIGURES)} manuscript figures in {_FIG_DIR}")
    return 0


def _environ() -> dict[str, str]:
    import os

    return os.environ.copy()


if __name__ == "__main__":
    raise SystemExit(main())
