#!/usr/bin/env python3
"""Regression tests for the SC-1 execution-stack gate bypass.

The Step 12 processor gates every rendered script before execution
(``execute.processor.execute_single_script``), but historically the
``GNNExecutor`` dispatch (shared by the MCP ``execute_gnn_model`` tool via
``execute_simulation_from_gnn`` and by the public ``gnn.execute`` API)
dispatched scripts to the per-framework runners without any pre-execution
scan — an ungated execution path. These tests pin the fix: the same
``scan_script_for_execution`` gate now runs at the ``GNNExecutor`` dispatch
and inside ``execute_script_safely`` (the shared runner choke point), so a
dangerous rendered script submitted through the executor stack is classified
``SecurityGateBlocked`` and never executed.

Note: the MCP tool itself only accepts GNN source files (``.md``/``.json``/
``.yaml``/``.yml``) and rejects ``.py`` payloads at path validation (pinned
by ``tests/execute/test_execute_mcp_wiring.py``). The gate therefore lives
on the ``GNNExecutor`` dispatch that the MCP tool delegates to, and on
``execute_script_safely`` which every Python framework runner funnels
through — together these close the "ungated remote-executable path".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute import execute_simulation_from_gnn  # noqa: E402
from gnn.execute.executor import (  # noqa: E402
    GNNExecutor,
    execute_gnn_model,
    execute_script_safely,
)

# ``run_subprocess_envelope`` sentinel: the process never started.
NEVER_STARTED = -1

_DANGEROUS_SCRIPT = "import os\nos.system('echo pwned')\n"
_BENIGN_SCRIPT = "x = [1, 2, 3]\nprint(sum(x))\n"


def _no_unsafe_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the escape hatch is off for the duration of a test."""
    monkeypatch.delenv("GNN_ALLOW_UNSAFE_EXEC", raising=False)


class TestExecutorGateBlocksDangerousScripts:
    """The GNNExecutor dispatch must not bypass the pre-execution gate."""

    def test_mcp_delegate_path_blocks_dangerous_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dangerous rendered script through the MCP delegation chain is
        classified SecurityGateBlocked and never executes."""
        sentinel = tmp_path / "sentinel"
        script = tmp_path / "dangerous_pymdp.py"
        script.write_text(
            f"import os\nos.system('touch {sentinel}')\n",
            encoding="utf-8",
        )
        _no_unsafe_exec(monkeypatch)

        result = execute_simulation_from_gnn(script, tmp_path / "exec_out")

        assert result["success"] is False
        assert result["error_type"] == "SecurityGateBlocked"
        assert "security gate blocked" in result["error"]
        assert result["security_findings"], "blocked findings must be recorded"
        assert not sentinel.exists(), "blocked script must never execute"

    def test_gnnexecutor_blocks_dangerous_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct GNNExecutor dispatch is gated (defense in depth)."""
        script = tmp_path / "dangerous.py"
        script.write_text(_DANGEROUS_SCRIPT)
        _no_unsafe_exec(monkeypatch)

        result = GNNExecutor(output_dir=str(tmp_path)).execute_gnn_model(
            str(script), execution_type="pymdp"
        )

        assert result["success"] is False
        assert result["error_type"] == "SecurityGateBlocked"
        assert any(
            finding.get("vulnerability_type") for finding in result["security_findings"]
        )

    def test_gate_block_classification_survives_convenience_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``execute_gnn_model`` convenience wrapper keeps the gate envelope."""
        script = tmp_path / "dangerous_jax.py"
        script.write_text("eval('__import__(\"os\").system(\"id\")')\n")
        _no_unsafe_exec(monkeypatch)

        result = execute_gnn_model(str(script), execution_type="jax")

        assert result["success"] is False
        assert result["error_type"] == "SecurityGateBlocked"
        assert result.get("status") == "FAILED"

    def test_escape_hatch_bypasses_executor_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GNN_ALLOW_UNSAFE_EXEC semantics match the Step 12 processor stack."""
        script = tmp_path / "dangerous.py"
        script.write_text("import os\nos.system('echo pwned')\n")
        monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

        result = GNNExecutor(output_dir=str(tmp_path)).execute_gnn_model(
            str(script), execution_type="pymdp"
        )

        # The gate must not be the reason for failure. The script may still
        # fail for unrelated reasons (imports); assert the classification
        # never appears.
        assert result.get("error_type") != "SecurityGateBlocked"


class TestExecutorPositivePath:
    """Benign scripts still execute through the gated dispatch."""

    def test_benign_script_executes_via_pymdp_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trivial no-op script runs offline via the pymdp runner path."""
        script = tmp_path / "benign_pymdp.py"
        script.write_text(_BENIGN_SCRIPT)
        _no_unsafe_exec(monkeypatch)

        result = GNNExecutor(output_dir=str(tmp_path)).execute_gnn_model(
            str(script), execution_type="pymdp", timeout=60
        )

        assert result["success"] is True, result.get("error", "")
        assert result["return_code"] == 0
        assert "6" in result["stdout"]

    def test_benign_script_executes_via_execute_script_safely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runner-level choke point passes benign scripts through."""
        script = tmp_path / "benign.py"
        script.write_text(_BENIGN_SCRIPT)
        _no_unsafe_exec(monkeypatch)

        result = execute_script_safely(script, timeout=60)

        assert result["success"] is True
        assert result["return_code"] == 0
        assert "6" in result["stdout"]

    def test_non_script_model_paths_skip_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GNN model files (.md) are not scripts; dispatch stays ungated."""
        model = tmp_path / "model.md"
        model.write_text("## ModelName\nsample\n")
        _no_unsafe_exec(monkeypatch)

        result = GNNExecutor(output_dir=str(tmp_path)).execute_gnn_model(
            str(model), execution_type="pymdp"
        )

        # Non-script inputs take the legacy "treated as source model" path.
        assert result["success"] is True
        assert "render/execute pipeline required" in result["stdout"]


class TestGateFailsClosed:
    """A broken security module blocks execution instead of skipping."""

    def test_gate_import_failure_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unimportable scanner -> SecurityGateBlocked (fail closed)."""
        script = tmp_path / "benign.py"
        script.write_text(_BENIGN_SCRIPT)
        _no_unsafe_exec(monkeypatch)
        monkeypatch.setitem(sys.modules, "gnn.security.processor", None)

        result = GNNExecutor(output_dir=str(tmp_path)).execute_gnn_model(
            str(script), execution_type="pymdp"
        )

        assert result["success"] is False
        assert result["error_type"] == "SecurityGateBlocked"
        assert "fail closed" in result["error"]

    def test_unreadable_script_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unscannable targets deny (deny_unreadable) and never start."""
        script = tmp_path / "missing.py"  # never created
        _no_unsafe_exec(monkeypatch)

        result = execute_script_safely(script)

        assert result["success"] is False
        assert result["error_type"] == "SecurityGateBlocked"
        assert result["return_code"] == NEVER_STARTED


def test_mcp_execute_path_result_shape_unchanged() -> None:
    """The MCP-visible envelope still carries the execution metadata keys.

    Guard against the gate accidentally reshaping the result the MCP tool
    returns: the block envelope is decorated with the same execution-time /
    device fields as a normal run.
    """
    executor = GNNExecutor(output_dir=str(Path(__file__).parent / "_scratch_shape"))
    blocked = executor.execute_gnn_model(
        "definitely_not_a_real_script.py", execution_type="pymdp"
    )
    assert blocked["success"] is False
    assert blocked["error_type"] == "SecurityGateBlocked"
    for key in ("execution_time", "execution_type", "model_path", "execution_device"):
        assert key in blocked, f"missing MCP envelope key: {key}"
