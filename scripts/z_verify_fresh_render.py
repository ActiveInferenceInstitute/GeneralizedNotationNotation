#!/usr/bin/env python3
"""Verify the fresh render against the committed custody manifest.

The step before recording in the scheduled re-render ritual (SC-22 — see
``scripts/z_generate_manuscript_variables.py``'s docstring): after the
template's ``stage_03_render`` has produced fresh PDF evidence and hydrated
prose, compare them to ``output/data/manuscript_render_manifest.json`` as
committed, BEFORE the record step overwrites it — so a red run certifies the
committed chain rather than just render success. ``[FAIL]`` means the
committed chain is stale for HEAD (an artifact is missing, or artifacts and
inputs both drifted — rerun the whole ritual); ``[WARN]`` means artifact-only
drift with the recorded inputs fully matching, i.e. toolchain variance the
v1 manifest cannot pin down (it records no tool versions).

Thin orchestrator, same shape as ``z_record_manuscript_render_manifest.py``:
all logic lives in :mod:`gnn.manuscript.render_custody`. Exits 1 iff any
``[FAIL]``; a ``[WARN]``-only run passes with the warnings on stderr.

Usage:
    uv run python scripts/z_verify_fresh_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    manifest_path,
    verify_fresh_render,
)


def main() -> int:
    issues = verify_fresh_render(_PROJECT_ROOT)
    for issue in issues:
        print(f"[render-custody] {issue}", file=sys.stderr)
    if any(issue.startswith("[FAIL]") for issue in issues):
        return 1
    if not issues:
        print(f"fresh render matches {manifest_path(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())