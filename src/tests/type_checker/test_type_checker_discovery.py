"""Discovery contract for the type checker (TO-DO: .gnn discovery gap).

The parser stack registers multiple spec extensions via
``get_supported_gnn_extensions()``, but the type checker's discovery used to
hardcode ``rglob("*.md")`` — a directory holding only a valid ``.gnn`` file
reported "No GNN files found". Discovery now walks every registered
non-binary extension; these tests pin that.
"""

from __future__ import annotations

from pathlib import Path

from gnn.parsers.common import get_supported_gnn_extensions
from type_checker.checking.core import GNNTypeChecker

MINIMAL_SPEC = """## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Discovery Probe

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=int]

## Connections
s-o

## Footer
Discovery Probe
"""


def test_supported_extensions_include_md_and_gnn() -> None:
    extensions = get_supported_gnn_extensions(include_binary_pickle=False)
    assert ".md" in extensions
    assert ".gnn" in extensions
    assert ".pickle" not in extensions


def test_type_checker_discovers_gnn_extension(tmp_path: Path) -> None:
    """A directory holding only a .gnn spec is discovered (not ignored)."""
    (tmp_path / "probe.gnn").write_text(MINIMAL_SPEC, encoding="utf-8")
    checker = GNNTypeChecker()
    discovered = checker._discover_gnn_files(tmp_path)
    assert [p.name for p in discovered] == ["probe.gnn"]


def test_type_checker_discovers_mixed_extensions(tmp_path: Path) -> None:
    """.md and .gnn specs in one directory are both discovered."""
    (tmp_path / "a.md").write_text(MINIMAL_SPEC, encoding="utf-8")
    (tmp_path / "b.gnn").write_text(MINIMAL_SPEC, encoding="utf-8")
    checker = GNNTypeChecker()
    discovered = checker._discover_gnn_files(tmp_path)
    assert sorted(p.name for p in discovered) == ["a.md", "b.gnn"]


def test_type_checker_ignores_binary_pickle_specs(tmp_path: Path) -> None:
    """.pickle specs are excluded from discovery — they are not type-checked."""
    (tmp_path / "probe.pickle").write_text(MINIMAL_SPEC, encoding="utf-8")
    checker = GNNTypeChecker()
    assert checker._discover_gnn_files(tmp_path) == []


def test_type_checker_ignores_repository_documentation(tmp_path: Path) -> None:
    """Common Markdown documentation names are not treated as model specs."""
    (tmp_path / "README.md").write_text(MINIMAL_SPEC, encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text(MINIMAL_SPEC, encoding="utf-8")
    (tmp_path / "probe.gnn").write_text(MINIMAL_SPEC, encoding="utf-8")

    checker = GNNTypeChecker()

    assert [path.name for path in checker._discover_gnn_files(tmp_path)] == [
        "probe.gnn"
    ]
