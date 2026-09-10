"""Default-suite contracts for zero-skip hardening work."""

from __future__ import annotations

import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SKIP_ALLOWLIST = {
    "tests/llm/test_llm_ollama.py",
    "tests/llm/test_llm_ollama_integration.py",
    "tests/analysis/test_rxinfer_cross_framework.py",
    "tests/pipeline/test_pomdp_gridworld_cross_framework.py",
    "tests/render/test_rxinfer_viz_log_contract.py",
    # Julia-live-backend gate: parse/execution tests skip when Julia or the
    # committed RxInfer environment is unavailable (same contract as the
    # GridWorld cross-framework and cross-framework-analysis files above).
    "tests/render/test_stigmergic_multi_agent.py",
    # Toolchain gates for backends outside the default lockfile: torch
    # (``uv sync --extra torch``; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47
    # but stays out of the default lock) and cmdstanpy/CmdStan (``uv sync
    # --extra stan`` + a compiled CmdStan). Rendering is always asserted;
    # only the execution half skips when the toolchain is absent.
    "tests/render/test_continuous_renderers.py",
    "tests/execute/test_execute_stan.py",
    # sklearn is an optional ``ml-ai`` extra; the inference round-trip tests
    # skip when scikit-learn is not installed (all uses are deferred imports).
    "tests/ml_integration/test_ml_integration_inference.py",
    # Bare-form evasions closed 2026-09-08 (deep horizon wave 2): the token
    # list below now also matches non-decorator ``pytest.mark.skip*`` and
    # ``unittest.skip*`` usage, so these previously invisible skip sites are
    # enumerated explicitly with their justifications.
    # Lean toolchain gate: fep_lean bridge tests skip when the ``lake``
    # binary/toolchain is unavailable (external toolchain, outside the lock).
    "tests/execute/test_lean_runner.py",
    # D2 module + ``d2`` system-binary gates (unittest.skipIf decorators).
    "tests/visualization/test_d2_visualizer.py",
    # Environment-integrity gates: skip rather than fail spuriously when the
    # JAX + pymdp dev stack is broken (``jax_pymdp_stack_ok()``).
    "tests/execute/test_execute_pymdp_simulation.py",
    "tests/pipeline/test_pomdp_pipeline_integration.py",
    # Permission probes: skip when running as root (the probe tests need a
    # non-root POSIX user for permission-based assertions).
    "tests/utils/test_shared_helpers.py",
    # Env opt-in gate (SC-44): ``test_uv_sync_fast`` runs
    # ``uv sync --frozen --check --inexact --extra dev``, which is
    # machine-dependent (wall-clock sensitive, racy against a concurrent
    # mutating sync on the shared ``.venv``). Skipped unless
    # ``GNN_UV_SYNC_LIVE=1`` — the same opt-in class as the Ollama live
    # files above.
    "tests/infrastructure/test_uv_sync_live.py",
}


FORBIDDEN_SKIP_TOKENS = (
    "pytest." + "skip(",
    "pytest." + "importorskip(",
    "pytest." + "xfail(",
    "@pytest.mark." + "skip",
    "@pytest.mark." + "skipif",
    "@pytest.mark." + "xfail",
    # Non-decorator marker forms (``pytestmark = pytest.mark.skipif(...)`` and
    # module-level marker variables) evade the @-prefixed tokens above.
    "pytest." + "mark.skip",
    "pytest." + "mark.xfail",
    # unittest-style skips: decorators and runtime raises.
    "unittest." + "skip(",
    "unittest." + "skipIf",
    "unittest." + "skipUnless",
)


def test_default_suite_does_not_reintroduce_skips_or_xfails() -> None:
    """The default suite must fail explicitly instead of hiding unavailable surfaces."""
    violations: list[str] = []
    for path in sorted((PROJECT_ROOT / "tests").rglob("test_*.py")):
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
    from gnn import parse_gnn_file
    from gnn.export import processor as export_processor

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
        for path in (PROJECT_ROOT / "tests").rglob("test_*.py")
        if path.name != Path(__file__).name
    )
    public_contracts = {
        "gnn.audio.processor._resolve_execution_summary_artifact": (
            "_resolve_execution_summary_artifact"
        ),
        "gnn.analysis.processor._scope_from_execution_summary": (
            "_scope_from_execution_summary"
        ),
        "gnn.export.processor.parse_gnn_content": "parse_gnn_content",
        "gnn.gui.gui_2.processor.run_gui": "static_headless_mode",
        "gnn.gui.gui_3.processor.run_gui": "static_headless_mode",
    }

    missing = [
        contract
        for contract, token in public_contracts.items()
        if token not in test_corpus
    ]
    assert not missing, f"Public contract ledger missing test references: {missing}"
