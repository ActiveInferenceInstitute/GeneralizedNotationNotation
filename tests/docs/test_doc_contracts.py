"""Regression tests for source-backed documentation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checker() -> Any:
    return load_module_from_path(
        "check_doc_contracts", REPO_ROOT / "scripts" / "check_doc_contracts.py"
    )


def test_repository_documentation_contracts_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checker = _load_checker()
    assert checker.scan() == []

    # Exercise refusal against independent fixture bytes. The inventory contract
    # must follow its source owners rather than freezing another numeric count.
    for relative in (
        "docs/gnn/tutorials/quickstart_tutorial.md",
        "docs/configuration/README.md",
        "docs/README.md",
        "docs/pipeline/README.md",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    assert checker.scan() == []
    pipeline = tmp_path / "docs/pipeline/README.md"
    original = pipeline.read_text(encoding="utf-8")
    for owner in (
        "src/gnn/render/framework_registry.py",
        "src/gnn/execute/executor_specs.py",
        "src/gnn/execute/detection.py",
    ):
        pipeline.write_text(original.replace(owner, "missing-owner"), encoding="utf-8")
        assert f"pipeline guide does not name inventory owner {owner}" in checker.scan()
    for term in ("experimental", "explicit"):
        pipeline.write_text(original.lower().replace(term, "omitted"), encoding="utf-8")
        assert (
            "pipeline guide does not explain explicit experimental selection"
            in checker.scan()
        )


def test_quickstart_lists_all_enforced_sections_in_order() -> None:
    checker = _load_checker()
    text = (
        REPO_ROOT / "docs" / "gnn" / "tutorials" / "quickstart_tutorial.md"
    ).read_text(encoding="utf-8")
    positions = [
        text.find(f"## {section}") for section in checker.REQUIRED_QUICKSTART_SECTIONS
    ]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
