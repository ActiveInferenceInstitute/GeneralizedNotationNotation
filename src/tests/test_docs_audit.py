"""Tests for doc/development/docs_audit.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_docs_audit():
    path = REPO_ROOT / "doc" / "development" / "docs_audit.py"
    spec = importlib.util.spec_from_file_location("docs_audit", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_doc_terms():
    path = REPO_ROOT / "scripts" / "check_maintained_doc_terms.py"
    spec = importlib.util.spec_from_file_location("check_maintained_doc_terms", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def format_strict_issue_detail():
    return _load_docs_audit().format_strict_issue_detail


def test_format_strict_issue_detail_lists_link_issues(
    format_strict_issue_detail,
) -> None:
    detail = format_strict_issue_detail(
        link_issues=[(Path("doc/x.md"), 10, "missing.md", "missing: doc/missing.md")],
        anchor_issues=[],
        anchor_checked=False,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
    )
    assert "Broken relative links" in detail
    assert "doc/x.md:10" in detail
    assert "`missing.md`" in detail


def test_format_strict_issue_detail_anchor_section_when_checked(
    format_strict_issue_detail,
) -> None:
    detail = format_strict_issue_detail(
        link_issues=[],
        anchor_issues=[
            (Path("doc/a.md"), 2, "b.md#frag", "anchor #frag not found (headings in `doc/b.md`)")
        ],
        anchor_checked=True,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
    )
    assert "Bad markdown anchors" in detail
    assert "doc/a.md:2" in detail


def test_maintained_doc_terms_flags_stale_phrase(tmp_path: Path) -> None:
    mod = _load_doc_terms()
    readme = tmp_path / "README.md"
    readme.write_text("This mentions FallbackAgent in maintained docs.\n", encoding="utf-8")

    findings = mod.scan(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path("README.md")
    assert "stale PyMDP" in findings[0].description


def test_maintained_doc_terms_skips_generated_and_archive_paths(tmp_path: Path) -> None:
    mod = _load_doc_terms()
    generated = tmp_path / "src" / "output"
    archived = tmp_path / "doc" / "archive"
    generated.mkdir(parents=True)
    archived.mkdir(parents=True)
    (generated / "README.md").write_text("FallbackAgent\n", encoding="utf-8")
    (archived / "old.md").write_text("simple_simulation.py\n", encoding="utf-8")

    findings = mod.scan(tmp_path)

    assert findings == []
