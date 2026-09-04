"""Contracts for ``tests.helpers``: script loader, GNN samples, MCP stubs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests.helpers import (
    SAMPLE_GNN_CONTENT,
    MCPTools,
    load_module_from_path,
    mcp_stubs,
    script_loader,
    write_sample_gnn_markdown,
)

pytestmark = pytest.mark.fast


def test_load_module_from_path_executes_and_registers(tmp_path: Path) -> None:
    script = tmp_path / "tiny_mod.py"
    script.write_text("VALUE = 41\n", encoding="utf-8")
    module = load_module_from_path("tiny_mod_under_test", script)
    assert module.VALUE == 41
    assert sys.modules.get("tiny_mod_under_test") is module


def test_load_module_from_path_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_module_from_path("nope", "/nonexistent/path/nope.py")


def test_load_module_from_path_sys_path_injection(tmp_path: Path) -> None:
    (tmp_path / "sibling_dep.py").write_text("N = 5\n", encoding="utf-8")
    script = tmp_path / "uses_sibling.py"
    script.write_text(
        "import sibling_dep\nTOTAL = sibling_dep.N * 2\n", encoding="utf-8"
    )
    module = load_module_from_path("uses_sibling_under_test", script, sys_path=tmp_path)
    assert module.TOTAL == 10


def test_load_module_from_path_removes_injected_sys_path(tmp_path: Path) -> None:
    marker = str(tmp_path)
    assert marker not in sys.path
    script = tmp_path / "plain.py"
    script.write_text("X = 1\n", encoding="utf-8")
    load_module_from_path("plain_under_test", script, sys_path=tmp_path)
    assert marker not in sys.path


def test_sample_gnn_content_has_required_sections() -> None:
    for section in ("## ModelName", "## StateSpaceBlock", "## Connections"):
        assert section in SAMPLE_GNN_CONTENT
    assert "test_model" in SAMPLE_GNN_CONTENT


def test_write_sample_gnn_markdown_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "actinf_pomdp_agent.md"
    write_sample_gnn_markdown(target)
    text = target.read_text(encoding="utf-8")
    assert "## ActInfOntologyAnnotation" in text
    assert "s = HiddenState" in text


def test_mcp_tools_positional_and_kwargs_registration() -> None:
    def tool_a(x: int) -> int:
        return x

    registry = MCPTools()
    registry.register_tool("a", tool_a, {"type": "object"}, "does a thing")
    registry.register_tool("b", function=tool_a, schema={}, description="kwarg form")
    assert registry.tools["a"]["function"] is tool_a
    assert registry.tools["a"]["schema"] == {"type": "object"}
    assert registry.tools["b"]["description"] == "kwarg form"
    assert registry.execute_tool("a", x=3) == 3


def test_mcp_tools_unknown_tool_returns_error_payload() -> None:
    registry = MCPTools()
    result: Any = registry.execute_tool("missing_tool", x=1)
    assert result == {"error": "tool_not_found", "name": "missing_tool"}


def test_mcp_tools_resources_and_helper_identity() -> None:
    registry = MCPTools()
    registry.register_resource("model://{name}", lambda name: name, "resolver")
    assert "model://{name}" in registry.resources
    assert mcp_stubs.MCPTools is MCPTools
    assert script_loader.load_module_from_path is load_module_from_path
