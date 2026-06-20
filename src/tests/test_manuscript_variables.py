"""Tests for the deterministic manuscript-variable producer.

Real objects only: every assertion recomputes the expected value from the live repository
and compares it against :func:`src.manuscript_variables.generate_variables`, so the
test fails if the producer drifts from the source surfaces it claims to read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.manuscript_variables import generate_variables, save_variables

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    return generate_variables(_PROJECT_ROOT)


def test_all_values_are_strings(variables: dict[str, str]) -> None:
    assert variables, "producer returned an empty token map"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in variables.items())


def test_token_keys_are_injector_compatible(variables: dict[str, str]) -> None:
    # Template injector regex: {{[A-Z][A-Z0-9_]*}}
    key_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    bad = [k for k in variables if not key_re.match(k)]
    assert not bad, f"keys not matchable by injector regex: {bad}"


def test_minimum_token_coverage(variables: dict[str, str]) -> None:
    # ISC-3: at least 20 distinct tokens.
    assert len(variables) >= 20, f"only {len(variables)} tokens"


def test_determinism(variables: dict[str, str]) -> None:
    # ISC-4: a second invocation yields an identical map.
    again = generate_variables(_PROJECT_ROOT)
    assert again == variables


def test_version_matches_pyproject(variables: dict[str, str]) -> None:
    text = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)
    assert variables["GNN_VERSION"] == expected


def test_step_count_matches_step_modules(variables: dict[str, str]) -> None:
    step_modules = list((_PROJECT_ROOT / "src").glob("[0-9]*_*.py"))
    assert variables["GNN_STEP_COUNT"] == str(len(step_modules))
    assert int(variables["GNN_STEP_COUNT"]) >= 1


def test_family_count_matches_manifest(variables: dict[str, str]) -> None:
    manifest = json.loads(
        (_PROJECT_ROOT / "input" / "model_family_manifest.json").read_text(encoding="utf-8")
    )
    assert variables["GNN_FAMILY_COUNT"] == str(len(manifest["families"]))


def test_backend_count_matches_registry(variables: dict[str, str]) -> None:
    registry = (_PROJECT_ROOT / "src" / "render" / "framework_registry.py").read_text(
        encoding="utf-8"
    )
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', registry)
    assert variables["GNN_BACKEND_COUNT"] == str(len(names))
    assert int(variables["GNN_BACKEND_COUNT"]) >= 2


def test_mcp_tool_count_matches_audit(variables: dict[str, str]) -> None:
    audit = json.loads(
        (_PROJECT_ROOT / "src" / "mcp" / "audit_report.json").read_text(encoding="utf-8")
    )
    assert variables["GNN_MCP_TOOL_COUNT"] == str(audit["tools_total"])


def test_tables_are_multiline_markdown(variables: dict[str, str]) -> None:
    for key in ("GNN_STEP_TABLE", "GNN_FAMILY_TABLE", "GNN_BACKEND_TABLE"):
        assert "\n" in variables[key], f"{key} should be a multi-line markdown table"
        assert variables[key].lstrip().startswith("|"), f"{key} should be a pipe table"


def test_license_value_is_resolved(variables: dict[str, str]) -> None:
    # ISC-23 guard: config completion removes the scaffold 'TBD'/empty license.
    assert variables["GNN_LICENSE"] and variables["GNN_LICENSE"] != "TBD"


def test_save_variables_roundtrip(tmp_path: Path, variables: dict[str, str]) -> None:
    out = save_variables(variables, tmp_path / "vars.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == variables
