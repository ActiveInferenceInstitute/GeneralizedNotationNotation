#!/usr/bin/env python3
"""Record the render custody manifest after a manuscript render.

The last step of the scheduled re-render ritual (SC-22 — see
``scripts/z_generate_manuscript_variables.py``'s docstring): after the
template's ``stage_03_render`` has produced the committed PDF evidence and
the producer has written the token map, digest both into
``output/data/manuscript_render_manifest.json`` so
``tests/test_manuscript_latex_log.py`` can prove the committed
``.log``/``.tex``/``.md``, the hydrated sections, and the token map are four
artifacts of one render rather than three renders passing together.

Thin orchestrator, same shape as ``z_generate_manuscript_variables.py``: all
logic lives in :mod:`gnn.manuscript.render_custody`. Refuses to record over
missing artifacts or a receipt/token-map commit disagreement.

Usage:
    uv run python scripts/z_record_manuscript_render_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript.render_custody import (  # noqa: E402
    manifest_path,
    record_render_manifest,
)


def main() -> int:
    try:
        record_render_manifest(_PROJECT_ROOT)
    except RuntimeError as exc:
        print(f"[render-custody] {exc}", file=sys.stderr)
        return 1
    print(manifest_path(_PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
