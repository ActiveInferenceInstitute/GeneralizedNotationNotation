"""Negative tests for the manuscript token gate's SC-3/SC-20 additions.

Each test plants the exact regression the gate rule exists for and asserts the
rule fires — plus the matching negative so a rule that fires on everything is
visible. Helpers are imported from the gate script by path (``scripts/`` is
not an importable package), the same convention
``tests/test_manuscript_path_claims.py`` uses.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_gate() -> types.ModuleType:
    """Import the gate script by path; ``scripts/`` is not an importable package."""
    script = REPO_ROOT / "scripts" / "check_manuscript_tokens.py"
    spec = importlib.util.spec_from_file_location("check_manuscript_tokens", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()

pytestmark: list[pytest.MarkDecorator] = [pytest.mark.fast, pytest.mark.unit]


# --- SC-20a: inverted step word orders ---------------------------------------


def test_inverted_25_steps_is_flagged() -> None:
    """Hard-typed "25 steps" must be flagged through the token-backed map."""
    step_literals = {"25": "GNN_STEP_COUNT"}
    issues = GATE._step_phrase_issues(
        "X.md", "The pipeline runs 25 steps.", step_literals
    )
    assert any("GNN_STEP_COUNT" in i and "25 steps" in i for i in issues), issues


def test_hyphenated_25_step_pipeline_is_flagged() -> None:
    issues = GATE._step_phrase_issues(
        "X.md",
        "the 25-step pipeline from parsing to reporting",
        {"25": "GNN_STEP_COUNT"},
    )
    assert any("GNN_STEP_COUNT" in i for i in issues), issues


def test_forward_step_phrase_still_flagged() -> None:
    issues = GATE._step_phrase_issues("X.md", "see step 3 below", {"3": "GNN_STEP_GNN"})
    assert any("GNN_STEP_GNN" in i for i in issues), issues


def test_unowned_step_number_is_not_flagged() -> None:
    """A number no token owns (someone else's count) stays clear."""
    assert (
        GATE._step_phrase_issues(
            "X.md", "the 30 steps of that survey", {"25": "GNN_STEP_COUNT"}
        )
        == []
    )


def test_token_backed_25_is_not_flagged() -> None:
    """{{GNN_STEP_COUNT}} is stripped before the scan, so the token is safe."""
    assert (
        GATE._step_phrase_issues(
            "X.md",
            "the {{GNN_STEP_COUNT}} steps of the pipeline",
            {"25": "GNN_STEP_COUNT"},
        )
        == []
    )


# --- SC-20c: crossref declarations -------------------------------------------


def test_typo_crossref_is_flagged(tmp_path: Path) -> None:
    section = tmp_path / "0X_section.md"
    section.write_text("See @fig:pipelinee for the DAG.\n", encoding="utf-8")
    declared = GATE._declared_labels([section], {})
    issues = GATE._dangling_xrefs({"fig:pipelinee"}, declared)
    assert issues and "fig:pipelinee" in issues[0]


def test_declared_crossref_passes(tmp_path: Path) -> None:
    section = tmp_path / "0X_section.md"
    section.write_text(
        "![cap](img.png){#fig:pipeline}\n\nSee @fig:pipeline.\n", encoding="utf-8"
    )
    declared = GATE._declared_labels([section], {})
    assert GATE._dangling_xrefs({"fig:pipeline"}, declared) == []


def test_producer_table_caption_counts_as_declaration(tmp_path: Path) -> None:
    """{#tbl:pipeline_steps} lives inside the producer's table token, not a .md."""
    variables = {
        "GNN_STEP_TABLE": (
            "| Step | Module |\n|---|---|\n| 3 | x |\n: cap {#tbl:pipeline_steps}"
        )
    }
    assert "tbl:pipeline_steps" in GATE._declared_labels([], variables)


# --- SC-3d: hydrated-copy unresolved-token scan ------------------------------


def test_unresolved_token_in_hydrated_copy_is_flagged(tmp_path: Path) -> None:
    hydrated = tmp_path / "output" / "manuscript"
    hydrated.mkdir(parents=True)
    (hydrated / "04_x.md").write_text(
        "The pipeline has {{GNN_TEST_FILE_COUNT}} files.\n", encoding="utf-8"
    )
    issues = GATE._hydrated_token_issues(hydrated)
    assert issues and "GNN_TEST_FILE_COUNT" in issues[0] and "04_x.md" in issues[0]


def test_documented_literal_braces_are_not_flagged(tmp_path: Path) -> None:
    """The shipped prose documents tokens as ``{{...}}`` — that must stay green."""
    hydrated = tmp_path / "output" / "manuscript"
    hydrated.mkdir(parents=True)
    (hydrated / "05_x.md").write_text(
        "write every claim as a double-brace `{{...}}` token.\n", encoding="utf-8"
    )
    assert GATE._hydrated_token_issues(hydrated) == []


def test_missing_hydrated_directory_is_not_an_issue(tmp_path: Path) -> None:
    assert GATE._hydrated_token_issues(tmp_path / "does_not_exist") == []


# --- SC-20d: registry filename validation ------------------------------------


def _registry_with_filename(tmp_path: Path, filename: str) -> Path:
    section = tmp_path / "0X_section.md"
    section.write_text("![d](f.png){#fig:x}\n", encoding="utf-8")
    fig_dir = tmp_path / "output" / "figures"
    fig_dir.mkdir(parents=True)
    (fig_dir / "figure_registry.json").write_text(
        json.dumps(
            {
                "figures": [
                    {"label": "fig:x", "alt_text": "desc", "filename": filename}
                ]
            }
        ),
        encoding="utf-8",
    )
    return section


def test_absolute_registry_filename_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted absolute path in figure_registry.json must not pass the join."""
    monkeypatch.setattr(GATE, "_PROJECT_ROOT", tmp_path)
    section = _registry_with_filename(tmp_path, "/etc/passwd")
    issues = GATE._figure_registry_issues(tmp_path / "manuscript", [section])
    assert any("/etc/passwd" in i and "bare filename" in i for i in issues), issues


def test_parent_escaping_registry_filename_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GATE, "_PROJECT_ROOT", tmp_path)
    section = _registry_with_filename(tmp_path, "../../secrets.png")
    issues = GATE._figure_registry_issues(tmp_path / "manuscript", [section])
    assert any("bare filename" in i for i in issues), issues


def test_normal_registry_filename_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GATE, "_PROJECT_ROOT", tmp_path)
    section = tmp_path / "0X_section.md"
    section.write_text("![d](f.png){#fig:x}\n", encoding="utf-8")
    fig_dir = tmp_path / "output" / "figures"
    fig_dir.mkdir(parents=True)
    (fig_dir / "real.png").write_bytes(b"png")
    (fig_dir / "figure_registry.json").write_text(
        json.dumps(
            {
                "figures": [
                    {"label": "fig:x", "alt_text": "desc", "filename": "real.png"}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert GATE._figure_registry_issues(tmp_path / "manuscript", [section]) == []


# --- SC-20b: hardcode target families ----------------------------------------


def test_figure_count_families_are_policed() -> None:
    """Figure-census counts at/above _HARDCODE_MIN are policed; small ones are not."""
    variables = {
        "GNN_OUTPUT_FIGURE_COUNT": "2000",
        "GNN_MANUSCRIPT_FIGURE_COUNT": "6",
        "GNN_MAINTAINED_FRAMEWORK_COUNT": "70",
    }
    targets = {
        variables[k]: k
        for k in (
            "GNN_OUTPUT_FIGURE_COUNT",
            "GNN_MANUSCRIPT_FIGURE_COUNT",
            "GNN_MAINTAINED_FRAMEWORK_COUNT",
        )
        if variables.get(k, "").isdigit() and int(variables[k]) >= GATE._HARDCODE_MIN
    }
    assert targets == {
        "2000": "GNN_OUTPUT_FIGURE_COUNT",
        "70": "GNN_MAINTAINED_FRAMEWORK_COUNT",
    }


def test_live_hardcode_policing_covers_the_new_families() -> None:
    """The gate's actual target map must police the new count families.

    Through the gate's own ``_hardcode_targets`` (not a replicated
    comprehension): the live >=_HARDCODE_MIN counts land in the map with
    their token names, and the sub-10 families stay out of the bare-number
    scan (phrase-anchored step detection covers those separately).
    """
    from gnn.manuscript.variables import generate_variables

    variables = generate_variables(REPO_ROOT)
    new_families = {
        "GNN_STEP_COUNT",
        "GNN_FAMILY_COUNT",
        "GNN_BACKEND_COUNT",
        "GNN_MAINTAINED_FRAMEWORK_COUNT",
        "GNN_OUTPUT_FIGURE_COUNT",
        "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT",
        "GNN_MANUSCRIPT_FIGURE_COUNT",
    }
    assert new_families.issubset(GATE._HARDCODE_KEYS), (
        new_families - set(GATE._HARDCODE_KEYS)
    )
    big = {k for k in new_families if int(variables[k]) >= GATE._HARDCODE_MIN}
    small = new_families - big
    targets = GATE._hardcode_targets(variables)
    for key in big:
        expected = variables[key]
        assert expected in targets, f"{key}={expected} is not policed"
        # GNN_OUTPUT_FIGURE_COUNT and GNN_OUTPUT_ARTIFACT_FIGURE_COUNT are
        # aliases of one census and share a value; the bare-number scan can
        # only attribute a literal to one of them, and either is correct.
        assert targets[expected] in {key, "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT"}, key
    for key in small:
        assert variables[key] not in targets, (
            f"{key}={variables[key]} is below _HARDCODE_MIN and must not be "
            "bare-number-policed (too coincidental)"
        )


# --- live end-to-end ----------------------------------------------------------


def test_live_gate_is_strict_green() -> None:
    """The gate must pass --strict on the committed corpus right now."""
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_manuscript_tokens.py"),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-500:]


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
