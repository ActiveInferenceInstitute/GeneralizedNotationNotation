#!/usr/bin/env python3
"""Hydrate GeneralizedNotationNotation manuscript variables before rendering.

Thin orchestrator (docxology/template contract). The render pipeline
(``infrastructure.rendering._manuscript_source.run_manuscript_variable_script``)
invokes this exact script name automatically before PDF render.

Responsibilities (delegated to :mod:`src.manuscript_variables`):

1. Compute the deterministic ``{{TOKEN}}`` map from the live repository.
2. Persist it to ``output/data/manuscript_variables.json`` for audit/debug.
3. Hydrate ``manuscript/*.md`` into ``output/manuscript/*.md`` with tokens
   resolved, via the template injector when available.

Runs standalone too (``python scripts/z_generate_manuscript_variables.py``);
the template hydration step is skipped gracefully when the sibling template
``infrastructure`` package is not importable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.manuscript_variables import generate_variables, save_variables  # noqa: E402

_INJECTION_REL = Path("infrastructure") / "rendering" / "manuscript_injection.py"


def _discover_template_root() -> Path | None:
    """Locate the sibling docxology/template checkout that owns the injector.

    The render launcher (``run_manuscript_variable_script``) runs this script with
    ``cwd=project_root`` and exports ``TEMPLATE_REPO_ROOT``. For a symlinked
    ``working/`` project the in-tree ``parents[2]`` trick does not reach the template
    root, so we resolve it from the env var first, then by walking up from both the
    invocation cwd (which may be the symlink path) and the project root.
    """
    candidates: list[Path] = []
    env_root = os.environ.get("TEMPLATE_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    for base in (Path.cwd(), Path(__file__).resolve().parent, _PROJECT_ROOT):
        candidates.append(base)
        candidates.extend(base.parents)
    for cand in candidates:
        if (cand / _INJECTION_REL).is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    return None


def main() -> int:
    variables = generate_variables(_PROJECT_ROOT)
    out_path = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    save_variables(variables, out_path)

    _discover_template_root()
    try:
        from infrastructure.rendering.manuscript_injection import (
            write_resolved_manuscript_tree,
        )
    except ModuleNotFoundError:
        print(
            "[manuscript-variables] template injector unavailable; "
            "wrote variables only (standalone mode)",
            file=sys.stderr,
        )
        print(str(out_path))
        return 0

    write_resolved_manuscript_tree(_PROJECT_ROOT, variables)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
