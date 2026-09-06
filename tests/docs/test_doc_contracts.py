"""Regression tests for source-backed documentation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checker() -> Any:
    return load_module_from_path(
        "check_doc_contracts", REPO_ROOT / "scripts" / "check_doc_contracts.py"
    )


def test_repository_documentation_contracts_pass() -> None:
    checker = _load_checker()
    assert checker.scan() == []


def test_quickstart_lists_all_enforced_sections_in_order() -> None:
    checker = _load_checker()
    text = (
        REPO_ROOT / "doc" / "gnn" / "tutorials" / "quickstart_tutorial.md"
    ).read_text(encoding="utf-8")
    positions = [
        text.find(f"## {section}") for section in checker.REQUIRED_QUICKSTART_SECTIONS
    ]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
