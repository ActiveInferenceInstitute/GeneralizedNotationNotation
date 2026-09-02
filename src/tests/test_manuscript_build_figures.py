"""Regression tests for ``scripts/manuscript_build_figures.py``.

The build orchestrator hard-codes a (generator, expected PNG) list with a
"keep in sync" comment. These tests lock that contract mechanically so a
new ``manuscript_fig_*.py`` generator that is not registered (or a stale
entry pointing at a removed generator) fails immediately instead of
silently producing an incomplete figure set.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "manuscript_build_figures.py"


def _figures_constant() -> list[tuple[str, str]]:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FIGURES" for t in node.targets
        ):
            entries: list[tuple[str, str]] = []
            assert isinstance(node.value, (ast.List, ast.Tuple))
            for item in node.value.elts:
                assert isinstance(item, ast.Tuple)
                gen, png = item.elts
                entries.append((ast.literal_eval(gen), ast.literal_eval(png)))
            return entries
    raise AssertionError("_FIGURES constant not found in manuscript_build_figures.py")


def test_every_generator_is_registered() -> None:
    """Every manuscript_fig_*.py generator appears in _FIGURES."""
    registered = {gen for gen, _ in _figures_constant()}
    on_disk = {p.name for p in (REPO_ROOT / "scripts").glob("manuscript_fig_*.py")}
    assert on_disk, "no manuscript figure generators found"
    missing = on_disk - registered
    assert not missing, f"generators missing from _FIGURES: {sorted(missing)}"


def test_no_stale_generator_entries() -> None:
    """Every registered generator and expected PNG name exists on disk."""
    for gen, png in _figures_constant():
        assert (REPO_ROOT / "scripts" / gen).is_file(), f"stale generator entry: {gen}"
        assert png.endswith(".png"), f"expected PNG must be .png: {png}"


def test_registered_generators_are_unique() -> None:
    """No duplicate generator registrations."""
    gens = [gen for gen, _ in _figures_constant()]
    assert len(gens) == len(set(gens))
