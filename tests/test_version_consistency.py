#!/usr/bin/env python3
"""SC-16 version consistency for ``src/gnn``.

The canonical version literal is the one in ``src/gnn/__init__.py``.
Subpackages re-declare module-metadata ``__version__`` constants; this scan
keeps every literal aligned with the canonical one so release bumps cannot
drift. Pure static check (AST scan of the repo tree) — no package import.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GNN_SRC = REPO_ROOT / "src" / "gnn"

# ``__version__`` declarations owned by another maintenance lane (execute/**
# is out of scope for the SC-16 cleanup) are tracked separately.
EXCLUDED_REL_PATHS: set[str] = {
    "src/gnn/execute/discopy_translator_module/__init__.py",
}


def _version_literals() -> dict[str, str]:
    """Map every ``__version__ = "<str>"`` assignment under ``src/gnn`` to its value."""
    literals: dict[str, str] = {}
    for path in sorted(GNN_SRC.rglob("*.py")):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if rel_path in EXCLUDED_REL_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__version__"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    literals[rel_path] = node.value.value
    return literals


def test_canonical_version_literal_exists() -> None:
    """The scan anchor must be present and the corpus non-trivial.

    The floor guards against a silently broken scan (REPO_ROOT resolution or
    rglob regression) passing vacuously on a near-empty literal map. Update
    it consciously if the single-source pattern changes.
    """
    literals = _version_literals()
    assert len(literals) >= 30, (
        f"version scan found only {len(literals)} literals; expected the "
        "known src/gnn corpus (~34) — REPO_ROOT resolution or the scan is broken"
    )
    assert "src/gnn/__init__.py" in literals


def test_all_version_literals_match_canonical() -> None:
    """Every ``__version__`` literal in src/gnn equals the canonical package version."""
    literals = _version_literals()
    canonical = literals["src/gnn/__init__.py"]
    assert canonical, "src/gnn/__init__.py must define a non-empty __version__ string"
    drifted = {
        rel_path: value for rel_path, value in literals.items() if value != canonical
    }
    assert not drifted, f"__version__ literals drifted from {canonical!r}: {drifted}"