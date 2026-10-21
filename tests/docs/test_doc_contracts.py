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
        "docs/INDEX.md",
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


def test_quickstart_authored_model_renders_through_real_pymdp_path(
    tmp_path: Path,
) -> None:
    """A syntax-valid tutorial must also survive scientific render admission."""
    import re

    import numpy as np

    from gnn.extract.pomdp_extractor import extract_pomdp_from_file
    from gnn.render.pomdp_processor import process_pomdp_for_frameworks

    text = (REPO_ROOT / "docs/gnn/tutorials/quickstart_tutorial.md").read_text()
    match = re.search(r"```gnn\n(.*?)\n```", text, re.DOTALL)
    assert match is not None
    source = tmp_path / "grid_agent.md"
    source.write_text(match[1] + "\n")
    model = extract_pomdp_from_file(source, strict_validation=True, on_error="raise")
    assert model is not None
    results = process_pomdp_for_frameworks(
        model, tmp_path / "render", frameworks=["pymdp"], gnn_file_path=source
    )
    assert results["overall_success"], results["framework_results"]
    assert list((tmp_path / "render").rglob("*_pymdp.py"))
    # The authored grid has four fixed actions; transitions preserve probability.
    b = np.asarray(model.B_matrix)
    assert b.shape == (4, 4, 4)
    assert np.allclose(b.sum(axis=0), 1.0)


def test_topic_index_requires_routes_to_existing_maintained_hubs(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    topic = tmp_path / "docs" / "new_topic"
    topic.mkdir(parents=True)
    (topic / "README.md").write_text("# New topic\n")
    index = tmp_path / "docs" / "INDEX.md"
    assert checker.missing_topic_routes(tmp_path) == ["documentation index is missing"]
    index.write_text("# Index\n")
    assert checker.missing_topic_routes(tmp_path) == [
        "documentation index is missing topic route new_topic/README.md"
    ]
    index.write_text("# Index\n[New topic](new_topic/README.md)\n")
    assert checker.missing_topic_routes(tmp_path) == []
