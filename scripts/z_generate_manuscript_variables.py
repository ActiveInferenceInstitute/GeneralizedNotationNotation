#!/usr/bin/env python3
"""Hydrate GeneralizedNotationNotation manuscript variables before rendering.

Thin orchestrator (docxology/template contract). The render pipeline
(``infrastructure.rendering._manuscript_source.run_manuscript_variable_script``)
invokes this exact script name automatically before PDF render.

Responsibilities (delegated to :mod:`gnn.manuscript_variables`):

1. Compute the deterministic ``{{TOKEN}}`` map from the repository at ``HEAD``.
2. Write the producer-owned ``manuscript/config.yaml`` fields (``version:``,
   ``date:``) from that map — config.yaml is never token-substituted, so those
   title-page literals can only stay true by being written.
3. Persist the map to ``output/data/manuscript_variables.json`` for audit/debug.
4. Hydrate ``manuscript/*.md`` into ``output/manuscript/*.md`` with tokens
   resolved, via the template injector when available.

Every count in the map describes the commit reported as ``GNN_GIT_COMMIT``. A
dirty working tree is reported on stderr because the rendered numbers will then
describe the last commit rather than what is on disk.

Runs standalone too (``python scripts/z_generate_manuscript_variables.py``);
the template hydration step is skipped gracefully when the sibling template
``infrastructure`` package is not importable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gnn.manuscript_variables import (  # noqa: E402
    generate_variables,
    save_variables,
    sync_config_metadata,
)

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


def _report_dirty_tree() -> None:
    """Say so when the working tree differs from the commit being counted."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return
    if result.returncode != 0 or not result.stdout.strip():
        return
    changed = len(result.stdout.strip().splitlines())
    print(
        f"[manuscript-variables] working tree has {changed} uncommitted change(s); "
        "every count describes the last commit, not the files on disk",
        file=sys.stderr,
    )


def main() -> int:
    variables = generate_variables(_PROJECT_ROOT)
    for change in sync_config_metadata(_PROJECT_ROOT, variables):
        print(f"[manuscript-variables] config.yaml {change}", file=sys.stderr)
    _report_dirty_tree()
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
