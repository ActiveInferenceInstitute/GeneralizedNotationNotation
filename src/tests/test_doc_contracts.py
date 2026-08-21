"""Regression tests for source-backed documentation contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checker() -> Any:
    path = REPO_ROOT / "scripts" / "check_doc_contracts.py"
    spec = importlib.util.spec_from_file_location("check_doc_contracts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
