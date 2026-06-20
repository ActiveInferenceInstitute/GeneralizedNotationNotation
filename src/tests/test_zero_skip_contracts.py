"""Default-suite contracts for zero-skip hardening work."""

from __future__ import annotations

import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_SKIP_ALLOWLIST = {
    "src/tests/llm/test_llm_ollama.py",
    "src/tests/llm/test_llm_ollama_integration.py",
}


FORBIDDEN_SKIP_TOKENS = (
    "pytest." + "skip(",
    "pytest." + "importorskip(",
    "pytest." + "xfail(",
    "@pytest.mark." + "skip",
    "@pytest.mark." + "skipif",
    "@pytest.mark." + "xfail",
)


def test_default_suite_does_not_reintroduce_skips_or_xfails() -> None:
    """The default suite must fail explicitly instead of hiding unavailable surfaces."""
    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / "src" / "tests").rglob("test_*.py")):
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        if relative_path in DEFAULT_SKIP_ALLOWLIST or path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SKIP_TOKENS:
            if token in text:
                violations.append(f"{relative_path}: contains {token}")

    assert not violations, "\n".join(violations)


def test_export_parse_gnn_content_reuses_canonical_gnn_parser() -> None:
    """Export's compatibility adapter must delegate parsing to ``gnn``."""
    from export import processor as export_processor
    from gnn import parse_gnn_file

    source = inspect.getsource(export_processor.parse_gnn_content)
    assert "parse_gnn_file" in source

    content = """# Parser Reuse Contract

## StateSpaceBlock
A[2,2,type=float]

## Connections
A -> B
"""
    export_data = export_processor.parse_gnn_content(content)
    canonical = parse_gnn_file("inline_export_input.md", content=content)

    assert export_data["canonical_parse"]["sections"] == canonical["sections"]
    assert export_data["canonical_parse"]["variables"] == canonical["variables"]
    assert export_data["canonical_parse"]["success"] is True
    assert isinstance(export_data["sections"], dict)
    assert isinstance(export_data["variables"], list)
    assert all(isinstance(variable, dict) for variable in export_data["variables"])


def test_public_contract_surface_ledger_is_covered_from_src_tests() -> None:
    """Critical public helpers touched by this pass have direct test references."""
    test_corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src" / "tests").rglob("test_*.py")
        if path.name != Path(__file__).name
    )
    public_contracts = {
        "audio.processor._resolve_execution_summary_artifact": (
            "_resolve_execution_summary_artifact"
        ),
        "analysis.processor._scope_from_execution_summary": (
            "_scope_from_execution_summary"
        ),
        "export.processor.parse_gnn_content": "parse_gnn_content",
        "gui.gui_2.processor.run_gui": "static_headless_mode",
        "gui.gui_3.processor.run_gui": "static_headless_mode",
    }

    missing = [
        contract
        for contract, token in public_contracts.items()
        if token not in test_corpus
    ]
    assert not missing, f"Public contract ledger missing test references: {missing}"
