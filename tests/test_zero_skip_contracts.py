"""Default-suite contracts for zero-skip hardening work."""

from __future__ import annotations

import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# S2-17 (2026-09-10): the former 14-entry allowlist is empty. Every file-local
# skipif/importorskip/unittest toolchain gate was migrated to registered
# ``needs_*`` markers (see pytest.ini) resolved dynamically by
# tests/conftest.py against tests/helpers/toolchain_probes.py. Any future
# skip site needs a marker migration, not an allowlist entry.
DEFAULT_SKIP_ALLOWLIST: set[str] = set()


# The one sanctioned dynamic-skip mechanism lives OUTSIDE the scanned corpus:
# tests/conftest.py applies ``pytest.mark.skip`` from the availability probes
# in tests/helpers/toolchain_probes.py for ``needs_*``-marked tests. Test
# files themselves must not contain any of the tokens below.
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
    "pytest." + "mark.skipif",
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


def _registered_marker_names() -> set[str]:
    """Parse the ``markers =`` block of pytest.ini into marker names."""
    names: set[str] = set()
    in_markers = False
    for line in (PROJECT_ROOT / "pytest.ini").read_text(encoding="utf-8").splitlines():
        if line == "markers =":
            in_markers = True
            continue
        if in_markers:
            if not line.startswith("    "):
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names.add(stripped.split(":", 1)[0].strip())
    return names


def test_needs_markers_pair_with_probes_and_registration() -> None:
    """Every ``needs_*`` marker is registered AND has a dynamic probe, and
    every probe-backed marker is registered (``--strict-markers`` would fail
    collection otherwise, but this pins the pytest.ini <-> probe mapping
    against silent drift in either file)."""
    from tests.helpers.toolchain_probes import TOOLCHAIN_MARKERS

    registered_needs = {
        name for name in _registered_marker_names() if name.startswith("needs_")
    }
    assert registered_needs == set(TOOLCHAIN_MARKERS), (
        f"pytest.ini needs_* markers and TOOLCHAIN_MARKERS diverged: "
        f"unprobed={sorted(registered_needs - set(TOOLCHAIN_MARKERS))} "
        f"unregistered={sorted(set(TOOLCHAIN_MARKERS) - registered_needs)}"
    )
    for name, (probe, reason) in TOOLCHAIN_MARKERS.items():
        assert callable(probe), f"{name}: probe is not callable"
        assert reason.strip(), f"{name}: empty skip reason"


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
