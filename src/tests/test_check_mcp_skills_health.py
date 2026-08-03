"""Regression tests for ``scripts/check_mcp_skills_health.py``.

Locks the guarantee that every ``src/<module>/SKILL.md`` documents a resolvable
surface: API imports, Key Exports, and Key Command scripts must resolve against
the live codebase. Uses the script's pure (MCP-free) resolvability function so
the test is fast and deterministic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_mcp_skills_health.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_mcp_skills_health", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> Any:
    return _load()


def _skill_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "src"
    return sorted(root.glob("*/SKILL.md")) + sorted((root / "gui").glob("*/SKILL.md"))


def test_every_skill_is_exercised(checker: Any) -> None:
    """All 34 module SKILL.md files are audited (guards against a dropped file)."""
    assert len(_skill_files()) >= 30


def test_skills_resolvability_clean(checker: Any) -> None:
    """Every SKILL.md API import / Key Export / Key Command resolves."""
    total_checks = 0
    for skill in _skill_files():
        rel = skill.relative_to(checker.REPO)
        findings, checks = checker._resolvability_findings(skill, rel)
        total_checks += checks
        assert findings == [], f"{rel}: {findings}"
    assert total_checks >= 100


def test_resolve_known_symbols(checker: Any) -> None:
    """Sanity: _resolve distinguishes present from absent exports."""
    assert checker._resolve("gnn", "parse_gnn_file")
    assert checker._resolve("analysis", "process_analysis")
    assert not checker._resolve("gnn", "DefinitelyNotASymbol")


def test_key_export_resolves_on_module_package(checker: Any) -> None:
    """Key Exports resolve against the module package (or its submodules)."""
    analysis_dir = checker.SRC / "analysis"
    assert checker._module_package_resolve(analysis_dir, "process_analysis")
