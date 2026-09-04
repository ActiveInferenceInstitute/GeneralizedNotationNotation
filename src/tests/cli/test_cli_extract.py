"""CLI `extract` subcommand and gnn.extract.extract_to_json contract.

Contract under test (pinned decision #8, impl-cli coordination):
- `gnn extract FILE` prints POMDPStateSpace.to_dict() JSON (via
  gnn.extract.extract_to_json) to stdout, exit 0 on success.
- Failure prints a {"status": "error", "error": {code, message, line, section}}
  envelope (error is an object) to stdout, exit 1.
- extract_to_json(path, *, strict_validation=True, on_error="lenient",
  compact=False) -> str with extraction_schema_version == "1.0.0".
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

try:
    import cli
    from cli import main
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import cli
    from cli import main

from gnn.extract import extract_to_json

REPO = Path(__file__).resolve().parents[3]
ACTINF_EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    """In-process main() invocation with captured stdout/stderr (test_cli.py pattern)."""
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    captured_out, captured_err = StringIO(), StringIO()
    sys.stdout, sys.stderr = captured_out, captured_err
    try:
        code = main(argv)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
    return code, captured_out.getvalue(), captured_err.getvalue()


def _parse_json_output(text: str) -> dict[str, Any]:
    """Parse the JSON payload, tolerating non-JSON preamble lines."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        brace = stripped.find("{")
        assert brace >= 0, f"no JSON payload found in output: {text!r}"
        parsed = json.loads(stripped[brace:])
    assert isinstance(parsed, dict)
    return parsed


def test_cli_extract_success_on_actinf_exemplar() -> None:
    code, out, _err = _run_main(["extract", str(ACTINF_EXEMPLAR)])
    assert code == 0, f"expected exit 0, got {code}; output: {out!r}"
    payload = _parse_json_output(out)
    assert payload["extraction_schema_version"] == "1.0.0"
    assert payload.get("model_name") == "Active Inference POMDP Agent"


def test_cli_extract_failure_envelope_on_non_gnn_file(tmp_path: Path) -> None:
    """A file extraction cannot handle -> exit 1 with a structured error envelope."""
    bad = tmp_path / "not_a_gnn.md"
    bad.write_text("# this is not a GNN model file\n")
    code, out, _err = _run_main(["extract", str(bad)])
    assert code == 1, f"expected exit 1, got {code}; output: {out!r}"
    envelope = _parse_json_output(out)
    assert envelope["status"] == "error"
    error = envelope["error"]
    assert isinstance(error, dict), "error must be an object, not a string"
    assert isinstance(error.get("code"), str) and error["code"]
    assert error.get("message")
    assert "line" in error and "section" in error


def test_extract_to_json_returns_serializable_schema_stamped_payload() -> None:
    raw = extract_to_json(ACTINF_EXEMPLAR)
    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert payload["extraction_schema_version"] == "1.0.0"
    assert "num_state_factors" in payload
    assert "dimension_provenance" in payload


def test_extract_to_json_compact_flag_produces_single_line() -> None:
    raw = extract_to_json(ACTINF_EXEMPLAR, compact=True)
    assert isinstance(raw, str)
    json.loads(raw)
    assert "\n" not in raw.strip()
