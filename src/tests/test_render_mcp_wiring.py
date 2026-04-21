"""
MCP wiring tests for ``src/render/mcp.py``.

Guards against signature / return-shape drift between the MCP tool shims and
the underlying render module API. Every registered tool must return a dict
with ``success`` and must not raise ``TypeError``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from render import mcp as render_mcp


class _CapturingMCP:
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
    render_mcp.register_tools(mcp)
    return {name: func for name, func, _schema, _desc in mcp.tools}


SAMPLE_GNN = (
    Path(__file__).parent.parent.parent
    / "input"
    / "gnn_files"
    / "discrete"
    / "actinf_pomdp_agent.md"
)


def test_register_tools_emits_expected_set(registered_tools: Dict[str, Any]) -> None:
    assert set(registered_tools) == {
        "process_render",
        "list_render_frameworks",
        "render_gnn_to_format",
        "get_render_module_info",
    }


def test_list_render_frameworks_shape(registered_tools: Dict[str, Any]) -> None:
    result = registered_tools["list_render_frameworks"]()
    assert isinstance(result, dict)
    assert "success" in result
    assert "frameworks" in result
    frameworks = result["frameworks"]
    assert isinstance(frameworks, dict), (
        "frameworks must be a dict for consistent consumer behavior; "
        f"got {type(frameworks).__name__}"
    )
    # Every entry must itself be a dict with ``available`` and ``description``.
    for name, entry in frameworks.items():
        assert isinstance(entry, dict), f"framework entry {name!r} must be dict"
        assert set(entry.keys()) >= {"available", "description"}, (
            f"framework entry {name!r} missing keys: {entry}"
        )
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["description"], str)


def test_get_render_module_info_shape(registered_tools: Dict[str, Any]) -> None:
    result = registered_tools["get_render_module_info"]()
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result.get("module", "").endswith("render")
    assert "frameworks" in result
    assert "input_formats" in result
    assert "output_formats" in result


def test_process_render_tool_handles_empty_directory(
    registered_tools: Dict[str, Any], tmp_path: Path
) -> None:
    empty_in = tmp_path / "empty_in"
    empty_in.mkdir()
    out_dir = tmp_path / "out"
    result = registered_tools["process_render"](str(empty_in), str(out_dir))
    assert isinstance(result, dict)
    assert isinstance(result.get("success"), bool)


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN not available")
def test_render_gnn_to_format_tool_accepts_real_gnn(
    registered_tools: Dict[str, Any], tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = registered_tools["render_gnn_to_format"](
        str(SAMPLE_GNN),
        str(out_dir),
        "pymdp",
    )
    assert isinstance(result, dict)
    assert "success" in result
    # Must not raise TypeError from bad wiring.
    assert isinstance(result["success"], bool)
