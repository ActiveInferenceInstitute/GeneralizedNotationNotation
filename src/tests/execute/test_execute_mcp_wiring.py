"""
MCP wiring tests for ``src/execute/mcp.py``.

These tests invoke each registered tool through the module-level MCP
registration path to guard against future signature drift between the MCP
tool shims and the underlying domain functions. They do not require a full
simulation to succeed: dependency-driven failures are allowed, but the tools
must always return a dictionary with a ``success`` key and must never raise
``TypeError`` due to bad argument wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execute import mcp as execute_mcp


class _CapturingMCP:
    """Minimal MCP instance that records the tools registered against it."""

    def __init__(self) -> None:
        self.tools: List[Tuple[str, Any, Dict[str, Any], str]] = []

    def register_tool(
        self,
        name: str,
        func: Any,
        schema: Dict[str, Any],
        description: str,
        **_kwargs: Any,
    ) -> None:
        self.tools.append((name, func, schema, description))


@pytest.fixture(scope="module")
def registered_tools() -> Dict[str, Any]:
    mcp = _CapturingMCP()
    execute_mcp.register_tools(mcp)
    return {name: func for name, func, _schema, _desc in mcp.tools}


@pytest.fixture(scope="module")
def registered_schemas() -> Dict[str, Dict[str, Any]]:
    mcp = _CapturingMCP()
    execute_mcp.register_tools(mcp)
    return {name: schema for name, _func, schema, _desc in mcp.tools}


SAMPLE_GNN = Path(__file__).parent.parent.parent.parent / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


def _assert_dict_result(result: Any) -> Dict[str, Any]:
    assert isinstance(result, dict), f"Tool must return a dict; got {type(result).__name__}"
    assert "success" in result, f"Tool result must include 'success': {result}"
    return result


def test_register_tools_emits_expected_set(registered_tools: Dict[str, Any]) -> None:
    expected = {
        "process_execute",
        "execute_gnn_model",
        "execute_pymdp_simulation",
        "check_execute_dependencies",
        "get_execute_module_info",
    }
    assert set(registered_tools) == expected, (
        f"Unexpected execute MCP tool set: {sorted(registered_tools)}"
    )


def test_schemas_match_tool_signatures(registered_schemas: Dict[str, Dict[str, Any]]) -> None:
    gnn_model_schema = registered_schemas["execute_gnn_model"]
    assert set(gnn_model_schema.get("properties", {}).keys()) == {
        "gnn_file_path",
        "output_directory",
    }
    pymdp_schema = registered_schemas["execute_pymdp_simulation"]
    assert set(pymdp_schema.get("properties", {}).keys()) == {
        "gnn_file_path",
        "output_directory",
    }


def test_check_execute_dependencies_returns_dict(registered_tools: Dict[str, Any]) -> None:
    result = _assert_dict_result(registered_tools["check_execute_dependencies"]())
    assert isinstance(result.get("success"), bool)


def test_get_execute_module_info_returns_metadata(registered_tools: Dict[str, Any]) -> None:
    result = _assert_dict_result(registered_tools["get_execute_module_info"]())
    assert result["success"] is True
    assert result.get("module", "").endswith("execute")
    assert "version" in result
    assert isinstance(result.get("tools"), list)


def test_process_execute_tool_handles_empty_directory(
    registered_tools: Dict[str, Any], tmp_path: Path
) -> None:
    # Name the directory with the 11_render_output suffix so the
    # processor's priority-2 heuristic stops at our empty dir instead of
    # falling back to real output/11_render_output globs (which would
    # run long-lived rendered scripts).
    empty_dir = tmp_path / "11_render_output"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"
    result = _assert_dict_result(
        registered_tools["process_execute"](str(empty_dir), str(out_dir))
    )
    # success may be False (no files) but the tool must not blow up.
    assert isinstance(result["success"], bool)


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN not available")
def test_execute_gnn_model_tool_accepts_real_gnn(
    registered_tools: Dict[str, Any], tmp_path: Path
) -> None:
    out_dir = tmp_path / "exec_out"
    result = _assert_dict_result(
        registered_tools["execute_gnn_model"](str(SAMPLE_GNN), str(out_dir))
    )
    # Must not raise TypeError from bad wiring; missing deps may yield success=False.
    assert "error" not in result or isinstance(result["error"], str)


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN not available")
def test_execute_pymdp_simulation_tool_accepts_real_gnn(
    registered_tools: Dict[str, Any], tmp_path: Path
) -> None:
    out_dir = tmp_path / "pymdp_out"
    result = _assert_dict_result(
        registered_tools["execute_pymdp_simulation"](str(SAMPLE_GNN), str(out_dir))
    )
    # Result shape: {"success": bool, ...merged simulator dict...}.
    # Tolerate dependency-driven failure, but a TypeError from mis-wiring
    # would have been raised before this assertion.
    assert isinstance(result["success"], bool)
