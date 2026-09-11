#!/usr/bin/env python3
"""Negative tests for SEC-R2 / S2-3: registry execution funnel and lean ``.md`` gate.

The registry runners (pymdp/jax/numpyro/pytorch/discopy) execute every rendered
script through ``execute_script_safely``, so the shared pre-execution security
gate must hold there, exactly as it does on the GNNExecutor dispatch path.
Lean dispatch executes ``.md`` documents through the fep-lean bridge
(``lean_runner.verify_document``), so a ``.md`` bound for lean is gate-checked
via its fenced code blocks while a ``.md`` parsed as data is not.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

import gnn.execute.executor as executor_module
from gnn.execute import execute_script_safely
from gnn.execute.executor import GNNExecutor, execute_rendered_simulators

# Would touch a marker file (proving execution) before the dangerous call the
# scanner must catch. Blocked-before-start keeps the marker absent.
_UNSAFE_SCRIPT = """import pathlib
pathlib.Path("marker.txt").write_text("ran")
import subprocess
subprocess.run("ls", shell=True)
"""


def _write_script(tmp_path: Path, name: str, content: str) -> Path:
    script = tmp_path / name
    script.write_text(content, encoding="utf-8")
    return script


def test_execute_script_safely_blocks_unsafe_rendered_script(tmp_path: Path) -> None:
    """The registry funnel refuses a dangerous rendered script before starting it."""
    script = _write_script(tmp_path, "unsafe_rendered.py", _UNSAFE_SCRIPT)

    result = execute_script_safely(script, cwd=tmp_path)

    assert result["success"] is False
    assert result["error_type"] == "SecurityGateBlocked"
    assert "Pre-execution security gate blocked" in result["error"]
    assert result["security_findings"], "block must carry the scanner findings"
    assert result["return_code"] == -1
    assert not (tmp_path / "marker.txt").exists(), "script must never have run"


def test_execute_script_safely_fail_closed_when_scanner_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable security scanner hard-blocks, never silently skips."""
    script = _write_script(tmp_path, "rendered.py", _UNSAFE_SCRIPT)
    monkeypatch.setitem(sys.modules, "gnn.security.processor", None)

    result = execute_script_safely(script, cwd=tmp_path)

    assert result["success"] is False
    assert result["error_type"] == "SecurityGateBlocked"
    assert "unavailable" in result["error"] or "scanner" in result["error"]
    assert not (tmp_path / "marker.txt").exists()


def test_execute_script_safely_carries_sandbox_envelope_keys(tmp_path: Path) -> None:
    """A clean script runs and reports GNN_SANDBOX semantics from the envelope."""
    script = _write_script(tmp_path, "ok.py", "print('hello')\n")

    result = execute_script_safely(script)

    assert result.get("error_type") != "SecurityGateBlocked"
    assert "hello" in result["stdout"]
    assert result["sandbox_mode"] == "off"
    assert result["sandboxed"] is False


def test_execute_rendered_simulators_registry_path_rejects_unsafe_script(
    tmp_path: Path,
) -> None:
    """The full registry entry rejects a rendered simulator the gate denies."""
    target = tmp_path / "rendered"
    pymdp_dir = target / "pymdp"
    pymdp_dir.mkdir(parents=True)
    unsafe = _write_script(pymdp_dir, "model_pymdp.py", _UNSAFE_SCRIPT)

    outcome = execute_rendered_simulators(
        target, tmp_path / "out", logging.getLogger("test-security-gate")
    )

    assert outcome is False
    assert not (pymdp_dir / "marker.txt").exists(), "unsafe script must never run"
    assert unsafe.is_file()


# --- lean ``.md`` gate (SEC-R3) -------------------------------------------------


def test_lean_md_with_dangerous_fence_is_gate_blocked(tmp_path: Path) -> None:
    """A lean-bound document with a dangerous fenced script is refused."""
    doc = _write_script(
        tmp_path,
        "model.md",
        "## GNNSection\n\n```python\n" + _UNSAFE_SCRIPT + "```\n",
    )

    result = GNNExecutor(output_dir=str(tmp_path / "out")).execute_gnn_model(
        str(doc), "lean"
    )

    assert result["success"] is False
    assert result["error_type"] == "SecurityGateBlocked"
    assert "Pre-execution security gate blocked" in result["error"]
    assert result["security_findings"]


def test_plain_gnn_md_is_not_gate_blocked_for_lean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GNN notation fences are data; a plain document is not denied by the gate."""
    doc = _write_script(
        tmp_path,
        "model.md",
        "## GNNSection\n\n```gnn\n### State\nS: {0.5, 0.5}\n```\n",
    )
    monkeypatch.setattr(executor_module, "LEAN_AVAILABLE", False)

    result = GNNExecutor(output_dir=str(tmp_path / "out")).execute_gnn_model(
        str(doc), "lean"
    )

    assert result.get("error_type") != "SecurityGateBlocked"


def test_md_under_non_lean_dispatch_is_not_gate_checked(tmp_path: Path) -> None:
    """A .md parsed only as data (non-lean dispatch) skips the executable scan."""
    doc = _write_script(tmp_path, "model.md", "## GNNSection\n[Fa]\n")

    result = GNNExecutor(output_dir=str(tmp_path / "out")).execute_gnn_model(
        str(doc), "pymdp"
    )

    assert result.get("error_type") != "SecurityGateBlocked"
