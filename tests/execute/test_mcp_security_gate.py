#!/usr/bin/env python3
"""
Negative tests for the pre-execution security gate on the GNNExecutor path.

SC-1: the MCP tool path (``execute_gnn_model_mcp`` →
``execute_simulation_from_gnn`` → ``GNNExecutor.execute_gnn_model``) runs
rendered scripts with no security scan. These tests pin the gate: an unsafe
rendered script submitted through the execute path returns
``SecurityGateBlocked``, never a run; and a sabotaged security import
hard-blocks (fail closed) instead of silently skipping the scan.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from gnn.execute.executor import GNNExecutor

_UNSAFE_SCRIPT = """import subprocess
subprocess.run("ls", shell=True)
"""

_SAFE_SCRIPT = "print('hello from a safe rendered script')\n"


def _write_script(tmp_path: Path, name: str, content: str) -> Path:
    script = tmp_path / name
    script.write_text(content, encoding="utf-8")
    return script


def _execute(executor: GNNExecutor, script: Path) -> dict[str, Any]:
    return executor.execute_gnn_model(str(script), "pymdp")


def test_unsafe_script_is_blocked_not_run(tmp_path: Path) -> None:
    """A rendered script with shell=True returns SecurityGateBlocked, not a run."""
    script = _write_script(tmp_path, "unsafe_rendered.py", _UNSAFE_SCRIPT)
    result = _execute(GNNExecutor(output_dir=str(tmp_path / "out")), script)

    assert result["success"] is False
    assert result["error_type"] == "SecurityGateBlocked"
    assert "Pre-execution security gate blocked" in result["error"]
    assert result["security_findings"], "block must carry the scanner findings"
    # The script must not have been executed: no attempt to start it.
    assert result.get("return_code") is None


def test_sabotaged_security_import_hard_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SC-2a: an unavailable security scanner fails CLOSED, never open."""
    script = _write_script(tmp_path, "rendered.py", _UNSAFE_SCRIPT)
    # sys.modules entry set to None makes `from gnn.security.processor import ...`
    # raise ImportError inside the gate helper.
    monkeypatch.setitem(sys.modules, "gnn.security.processor", None)

    result = _execute(GNNExecutor(output_dir=str(tmp_path / "out")), script)

    assert result["success"] is False
    assert result["error_type"] == "SecurityGateBlocked"
    assert "unavailable" in result["error"] or "scanner" in result["error"]


def test_allow_unsafe_exec_opt_out_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only way past the gate is the explicit GNN_ALLOW_UNSAFE_EXEC opt-out."""
    script = _write_script(tmp_path, "unsafe_rendered.py", _UNSAFE_SCRIPT)
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    result = _execute(GNNExecutor(output_dir=str(tmp_path / "out")), script)

    # Unsafe script was actually attempted: any non-gate outcome (subprocess
    # envelope success or failure) proves the opt-out bypassed the gate.
    assert result.get("error_type") != "SecurityGateBlocked"


def test_safe_script_passes_gate(tmp_path: Path) -> None:
    """A clean rendered script is not blocked by the gate."""
    script = _write_script(tmp_path, "safe_rendered.py", _SAFE_SCRIPT)
    result = _execute(GNNExecutor(output_dir=str(tmp_path / "out")), script)

    assert result.get("error_type") != "SecurityGateBlocked"
    assert result.get("success") in (True, False)  # execution attempted


def test_non_python_source_path_is_not_gate_blocked(tmp_path: Path) -> None:
    """Model-source passthrough (non-script paths) must not trip the scanner."""
    src = _write_script(tmp_path, "model.md", "# A GNN spec\n\n[Foo]\n")
    result = _execute(GNNExecutor(output_dir=str(tmp_path / "out")), src)

    assert result.get("error_type") != "SecurityGateBlocked"
