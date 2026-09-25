#!/usr/bin/env python3
"""
Gated-envelope pin for the MCP PyMDP execution path (M-12).

``execute_pymdp_simulation_mcp`` must run the simulation through the shared
subprocess envelope (``run_subprocess_envelope``) — the same gate every other
execute backend honors — instead of executing the model in-process. This test
pins the gate without executing anything: under ``GNN_SANDBOX=require`` with
no sandbox backend available, the envelope refuses pre-spawn, and the tool
must surface that refusal as a structured failure receipt rather than run
the simulation unsandboxed (the in-process path ignored ``GNN_SANDBOX``
entirely).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Generator

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute import mcp as execute_mcp
from gnn.execute.subprocess_envelope import NEVER_STARTED

SAMPLE_GNN = (
    Path(__file__).parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


@pytest.fixture
def repo_output_scratch(request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    base = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "test_mcp_pymdp_envelope"
        / request.node.name
    )
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_mcp_pymdp_path_honors_sandbox_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_output_scratch: Path,
) -> None:
    """GNN_SANDBOX=require with no backend refuses the MCP PyMDP path pre-spawn."""
    # No sandbox backend on this host, whatever is actually installed.
    monkeypatch.setattr("gnn.execute.sandbox.detect_sandbox", lambda: None)
    monkeypatch.setenv("GNN_SANDBOX", "require")

    out_dir = repo_output_scratch / "pymdp_out"
    result: dict[str, Any] = execute_mcp.execute_pymdp_simulation_mcp(
        str(SAMPLE_GNN), str(out_dir)
    )

    assert isinstance(result, dict), f"tool must return a dict; got {type(result)}"
    assert result["success"] is False
    assert result["error_type"] == "SandboxUnavailable"
    assert result["return_code"] == NEVER_STARTED
    assert "sandbox" in result["error"].lower()
    # The gate refused pre-spawn: no child process ran, so the model was
    # never executed and no artifacts were produced.
    assert not any(out_dir.iterdir()), (
        f"refused run must not execute the model; found {list(out_dir.iterdir())}"
    )
