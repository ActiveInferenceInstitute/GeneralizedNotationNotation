"""Tests for docs/development/docs_audit.py helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_docs_audit() -> Any:
    return load_module_from_path(
        "docs_audit", REPO_ROOT / "docs" / "development" / "docs_audit.py"
    )


def _load_doc_terms() -> Any:
    return load_module_from_path(
        "check_maintained_doc_terms",
        REPO_ROOT / "scripts" / "check_maintained_doc_terms.py",
    )


@pytest.fixture()
def format_strict_issue_detail() -> Any:
    return _load_docs_audit().format_strict_issue_detail


def test_format_strict_issue_detail_lists_link_issues(
    format_strict_issue_detail: Any,
) -> None:
    detail = format_strict_issue_detail(
        link_issues=[(Path("docs/x.md"), 10, "missing.md", "missing: docs/missing.md")],
        anchor_issues=[],
        anchor_checked=False,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
        security_version_issues=[],
        spec_coverage_issues=[],
        prose_src_issues=[],
        version_claim_issues=[],
    )
    assert "Broken relative links" in detail
    assert "docs/x.md:10" in detail
    assert "`missing.md`" in detail


def test_format_strict_issue_detail_anchor_section_when_checked(
    format_strict_issue_detail: Any,
) -> None:
    detail = format_strict_issue_detail(
        link_issues=[],
        anchor_issues=[
            (
                Path("docs/a.md"),
                2,
                "b.md#frag",
                "anchor #frag not found (headings in `docs/b.md`)",
            )
        ],
        anchor_checked=True,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
        security_version_issues=[],
        spec_coverage_issues=[],
        prose_src_issues=[],
        version_claim_issues=[],
    )
    assert "Bad markdown anchors" in detail
    assert "docs/a.md:2" in detail


# ---------------------------------------------------------------------------
# BC-17 checks: SPEC coverage, prose src/<module> patterns, version claims.
# Each test seeds a violation in a synthetic tree and proves the check FAILS
# on it (never vacuous), plus the canonical-form pass case.
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_mod(tmp_path: Path) -> Any:
    mod = _load_docs_audit()
    original_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    yield mod
    mod.REPO_ROOT = original_root


def test_spec_coverage_flags_module_without_spec(
    audit_mod: Any, tmp_path: Path
) -> None:
    module = tmp_path / "src" / "gnn" / "widget"
    module.mkdir(parents=True)
    (module / "core.py").write_text("x = 1\n", encoding="utf-8")

    missing = audit_mod.audit_src_spec_coverage()

    assert (
        Path("src/gnn/widget"),
        "module directory has .py files but no SPEC.md",
    ) in [(rel, msg) for rel, msg in missing]


def test_spec_coverage_passes_when_spec_present(audit_mod: Any, tmp_path: Path) -> None:
    module = tmp_path / "src" / "gnn" / "widget"
    module.mkdir(parents=True)
    (module / "core.py").write_text("x = 1\n", encoding="utf-8")
    (module / "SPEC.md").write_text("# Widget spec\n", encoding="utf-8")

    assert audit_mod.audit_src_spec_coverage() == []


def test_spec_coverage_ignores_py_free_dirs(audit_mod: Any, tmp_path: Path) -> None:
    module = tmp_path / "src" / "gnn" / "assets"
    module.mkdir(parents=True)
    (module / "data.bin").write_bytes(b"\x00")

    assert audit_mod.audit_src_spec_coverage() == []


def test_prose_pattern_flags_removed_layout_path(
    audit_mod: Any, tmp_path: Path
) -> None:
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "The MCP tools live in src/gui/mcp.py per the guide.\n"
        "See also import src.main in the tutorial prose.\n"
        "Canonical form: src/gnn/gui/ and gnn.main are fine.\n"
        "Quoted `src/gui/mcp.py` and `import src.main` are exempt.\n",
        encoding="utf-8",
    )

    issues = audit_mod.audit_prose_src_patterns([doc])

    flagged = {reason for _, _, reason in issues}
    assert any(
        "src/gui/" in r and "not the canonical src/gnn/ surface" in r for r in flagged
    )
    assert any("src.main" in r for r in flagged)
    assert len(issues) == 2


def test_prose_pattern_ignores_code_spans_and_canonical(
    audit_mod: Any, tmp_path: Path
) -> None:
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "- `src/gui/mcp.py` quoted path is fine\n"
        "- plain src/gnn/gui/mcp.py is canonical\n",
        encoding="utf-8",
    )

    assert audit_mod.audit_prose_src_patterns([doc]) == []


def _seed_pyproject(tmp_path: Path, version: str) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
    )


def test_version_claims_flag_stale_pipeline_label(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_pyproject(tmp_path, "3.5.0")
    doc = tmp_path / "docs" / "guide.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "Pipeline Version: 3.2.0\nCurrent Version: 1.0.0\n", encoding="utf-8"
    )

    issues = audit_mod.audit_version_claims([doc])

    assert len(issues) == 2
    assert issues[0][2] == "3.2.0"
    assert issues[1][2] == "1.0.0"


def test_version_claims_enforce_bare_version_only_in_package_scope(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_pyproject(tmp_path, "3.5.0")
    src_doc = tmp_path / "src" / "gnn" / "mod" / "SPEC.md"
    src_doc.parent.mkdir(parents=True)
    src_doc.write_text("**Version**: 1.2.3\n", encoding="utf-8")
    docs_doc = tmp_path / "docs" / "paper.md"
    docs_doc.parent.mkdir(parents=True)
    docs_doc.write_text("Version: 1.2.3\n", encoding="utf-8")

    src_issues = audit_mod.audit_version_claims([src_doc])
    docs_issues = audit_mod.audit_version_claims([docs_doc])

    assert len(src_issues) == 1  # package-version label scope
    assert docs_issues == []  # docs/** bare Version is a revision axis


def test_version_claims_pass_canonical_and_current(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_pyproject(tmp_path, "3.5.0")
    doc = tmp_path / "AGENTS.md"
    doc.write_text(
        "**Pipeline Version**: [pyproject.toml](pyproject.toml) (canonical)\n"
        "### Current Version: [pyproject.toml](pyproject.toml) (canonical)\n"
        "**Version**: [pyproject.toml](pyproject.toml) (canonical)\n",
        encoding="utf-8",
    )

    assert audit_mod.audit_version_claims([doc]) == []


def test_version_claims_pass_matching_current_version(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_pyproject(tmp_path, "3.5.0")
    doc = tmp_path / "src" / "gnn" / "mod" / "AGENTS.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("### Current Version: 3.5.0\n**Version**: 3.5.0\n", encoding="utf-8")

    assert audit_mod.audit_version_claims([doc]) == []


def _seed_framework_registry(tmp_path: Path, count: int) -> None:
    registry = tmp_path / "src" / "gnn" / "render" / "framework_registry.py"
    registry.parent.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(f'    "fw{i}": {{"name": "Fw{i}"}},' for i in range(count))
    registry.write_text(
        "FRAMEWORK_REGISTRY = {\n" + entries + "\n}\n", encoding="utf-8"
    )


def test_engine_count_flags_stale_footer(audit_mod: Any, tmp_path: Path) -> None:
    _seed_framework_registry(tmp_path, 3)
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "**Modules**: 7 · **Renderers**: 4 backends (see [x](x))\n",
        encoding="utf-8",
    )

    issues = audit_mod.audit_engine_count_claims([doc])

    assert len(issues) == 1
    assert issues[0][0] == Path("docs/note.md")
    assert issues[0][2] == "4"
    assert "framework_registry" in issues[0][3]


def test_engine_count_passes_matching_footer(audit_mod: Any, tmp_path: Path) -> None:
    _seed_framework_registry(tmp_path, 3)
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("**Renderers**: 3 backends (see [x](x))\n", encoding="utf-8")

    assert audit_mod.audit_engine_count_claims([doc]) == []


def test_engine_count_flags_registry_bound_prose(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_framework_registry(tmp_path, 3)
    doc = tmp_path / "SPEC.md"
    doc.write_text(
        "the four computational engines declared in x: A, B\n"
        "single declaration of the four frameworks\n"
        "Supports five backends: A, B\n",
        encoding="utf-8",
    )

    issues = audit_mod.audit_engine_count_claims([doc])

    assert [i[2] for i in issues] == ["four", "four", "five"]


def test_engine_count_passes_subset_and_idiom_phrases(
    audit_mod: Any, tmp_path: Path
) -> None:
    _seed_framework_registry(tmp_path, 3)
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "reports it alongside the other two backends\n"
        "Step 12 backends remain core deps\n"
        "renders one parsed spec to four backends — A, B\n"
        "the dispatch covers all five backends\n",
        encoding="utf-8",
    )

    assert audit_mod.audit_engine_count_claims([doc]) == []


def test_engine_count_covers_root_spec(audit_mod: Any, tmp_path: Path) -> None:
    _seed_framework_registry(tmp_path, 3)
    spec = tmp_path / "SPEC.md"
    spec.write_text(
        "the four computational engines declared in x: A, B\n", encoding="utf-8"
    )

    assert spec in audit_mod.engine_count_scan_files()
    issues = audit_mod.audit_engine_count_claims([spec])

    assert len(issues) == 1
    assert issues[0][2] == "four"


def test_engine_count_reports_unreadable_registry(
    audit_mod: Any, tmp_path: Path
) -> None:
    doc = tmp_path / "docs" / "note.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("**Renderers**: 4 backends\n", encoding="utf-8")

    issues = audit_mod.audit_engine_count_claims([doc])

    assert len(issues) == 1
    assert issues[0][3] == "cannot parse FRAMEWORK_REGISTRY"


def test_format_strict_issue_detail_lists_engine_count_claims() -> None:
    detail = _load_docs_audit().format_strict_issue_detail(
        link_issues=[],
        anchor_issues=[],
        anchor_checked=True,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
        security_version_issues=[],
        spec_coverage_issues=[],
        prose_src_issues=[],
        version_claim_issues=[],
        engine_count_issues=[
            (
                Path("docs/note.md"),
                3,
                "9",
                "render-engine count claim != framework_registry (10 frameworks)",
            ),
        ],
    )

    assert "Render-engine count claims (1)" in detail
    assert "docs/note.md:3" in detail


def test_format_strict_issue_detail_lists_bc17_sections() -> None:
    detail = _load_docs_audit().format_strict_issue_detail(
        link_issues=[],
        anchor_issues=[],
        anchor_checked=False,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
        security_version_issues=[],
        spec_coverage_issues=[
            (Path("src/gnn/widget"), "module directory has .py files but no SPEC.md")
        ],
        prose_src_issues=[
            (
                Path("docs/n.md"),
                3,
                "prose import 'src.main' is not the canonical gnn.* surface",
            )
        ],
        version_claim_issues=[
            (Path("AGENTS.md"), 632, "3.3.0", "version claim != pyproject 3.5.0")
        ],
    )
    assert "src/ dirs missing SPEC.md" in detail
    assert "src/gnn/widget" in detail
    assert "Prose src/<module>/ pattern" in detail
    assert "docs/n.md:3" in detail
    assert "Stale version claims" in detail
    assert "3.3.0" in detail


def test_format_strict_issue_detail_bc17_empty_by_default() -> None:
    detail = _load_docs_audit().format_strict_issue_detail(
        link_issues=[],
        anchor_issues=[],
        anchor_checked=False,
        spec_issues=[],
        coverage=[],
        doc_missing_agents=[],
        doc_missing_readme=[],
        agents_no_readme=[],
        readme_no_agents=[],
        doc_agents_structure=[],
        security_version_issues=[],
        spec_coverage_issues=[],
        prose_src_issues=[],
        version_claim_issues=[],
    )
    assert "src/ dirs missing SPEC.md" not in detail
    assert "Prose src/<module>/ pattern" not in detail
    assert "Stale version claims" not in detail


def test_maintained_doc_terms_flags_stale_phrase(tmp_path: Path) -> None:
    mod = _load_doc_terms()
    readme = tmp_path / "README.md"
    readme.write_text(
        "This mentions FallbackAgent in maintained docs.\n", encoding="utf-8"
    )

    findings = mod.scan(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path("README.md")
    assert "stale PyMDP" in findings[0].description


def test_maintained_doc_terms_skips_generated_and_archive_paths(tmp_path: Path) -> None:
    mod = _load_doc_terms()
    generated = tmp_path / "src" / "output"
    archived = tmp_path / "docs" / "other"
    generated.mkdir(parents=True)
    archived.mkdir(parents=True)
    (generated / "README.md").write_text("FallbackAgent\n", encoding="utf-8")
    (archived / "old.md").write_text("simple_simulation.py\n", encoding="utf-8")

    findings = mod.scan(tmp_path)

    assert findings == []
