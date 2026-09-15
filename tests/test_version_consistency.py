"""Version single-source invariant (residual of closed PR #106).

The package version must be defined exactly once, as a string literal,
somewhere under ``src/gnn/**/*.py`` (today that is ``src/gnn/__init__.py``),
and ``pyproject.toml`` must carry the same value. Every other module
re-exports the canonical ``gnn.__version__`` (``from gnn import
__version__``); a stray second literal — e.g. a copy-pasted
``__version__ = "..."`` in a submodule — must fail here instead of
silently diverging from the packaging metadata.

Deliberately NOT pinned: prose surfaces that quote the version (README,
TO-DO, docs/VERSION_MAP.md, per-module AGENTS.md). Those update at
release time and are not factually coupled to the literal; pinning them
here would turn every release bump into a test edit.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "gnn"


def _version_literals() -> list[tuple[Path, int, str]]:
    """Collect every ``__version__ = "..."`` string literal in the package.

    Returns ``(relative path, line number, literal value)`` triples, in
    deterministic (path-sorted, AST) order. Re-exports such as
    ``from gnn import __version__`` and plain ``"__version__"`` strings in
    ``__all__`` lists do not count: only assignments whose target is the
    name ``__version__`` and whose value is a string constant.
    """
    found: list[tuple[Path, int, str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in targets
            ):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.append((path.relative_to(REPO_ROOT), node.lineno, value.value))
    return found


def _sole_version_literal() -> str:
    """Return the package's single ``__version__`` literal, or fail loudly."""
    literals = _version_literals()
    if len(literals) != 1:
        rendered = ", ".join(
            f"{path}:{line} -> {value!r}" for path, line, value in literals
        )
        pytest.fail(
            f"expected exactly one __version__ string literal under "
            f"src/gnn/**/*.py, found {len(literals)}: {rendered or '(none)'}"
        )
    return literals[0][2]


def test_exactly_one_version_literal_in_package() -> None:
    literals = _version_literals()
    rendered = ", ".join(
        f"{path}:{line} -> {value!r}" for path, line, value in literals
    )
    assert len(literals) == 1, (
        f"expected exactly one __version__ string literal under "
        f"src/gnn/**/*.py, found {len(literals)}: {rendered or '(none)'}"
    )


def test_pyproject_version_matches_package_literal() -> None:
    expected = _sole_version_literal()
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    assert pyproject["project"]["version"] == expected
