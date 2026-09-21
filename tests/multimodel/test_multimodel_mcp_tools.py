#!/usr/bin/env python3
"""
Exercises the multimodel module MCP tool handlers.

Covers the dependency-graph rendering tool end to end: success rendering
against a repo exemplar and a constructed multi-model file, the typed
error paths (missing file, invalid output_format), and the MCP
registration shape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.multimodel import mcp as multimodel_mcp

REPO_ROOT = Path(__file__).resolve().parents[2]

MULTI_MODEL_GNN = """\
## StateSpaceBlock
# Agent 1
a[2,2,type=float]

## Connections
- a -> a

---
## StateSpaceBlock
# Agent 2
b[3,3,type=float]
a[2,2,type=float]

## Connections
- b -> b
"""


class _FakeRegistry:
    """Minimal recording registry matching the MCP ``register_tool`` surface."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        # Positional audio-style calls: (name, fn, schema, description)
        record: dict[str, Any] = dict(kwargs)
        for i, key in enumerate(("name", "function", "schema", "description")):
            if i < len(args):
                record[key] = args[i]
        self.registered.append(record)


class TestGenerateDependencyGraph:
    """Tests for the generate_dependency_graph_mcp wrapper."""

    @pytest.mark.unit
    def test_happy_path_repo_exemplar(self) -> None:
        """A real repo GNN file renders a non-empty graph and echoes inputs."""
        exemplar = REPO_ROOT / "input/gnn_files/multiagent/multi_agent_coordination.md"
        assert exemplar.is_file(), f"repo exemplar missing: {exemplar}"
        result = multimodel_mcp.generate_dependency_graph_mcp(str(exemplar))
        assert result["success"] is True
        assert result["file_path"] == str(exemplar)
        assert result["format"] == "mermaid"
        assert isinstance(result["graph"], str)
        assert result["graph"].strip(), "expected a non-empty rendered graph"

    @pytest.mark.unit
    def test_multi_model_file_renders_shared_edge(self, tmp_path: Any) -> None:
        """A two-model file yields two nodes and a shared-variable edge."""
        gnn = tmp_path / "two_models.md"
        gnn.write_text(MULTI_MODEL_GNN)
        result = multimodel_mcp.generate_dependency_graph_mcp(str(gnn))
        assert result["success"] is True
        assert result["format"] == "mermaid"
        graph = result["graph"]
        assert "Model_0" in graph
        assert "Model_1" in graph
        assert "shared: a" in graph

    @pytest.mark.unit
    def test_text_format_echoed(self, tmp_path: Any) -> None:
        """output_format 'text' renders the adjacency list and is echoed."""
        gnn = tmp_path / "two_models.md"
        gnn.write_text(MULTI_MODEL_GNN)
        result = multimodel_mcp.generate_dependency_graph_mcp(
            str(gnn), output_format="text"
        )
        assert result["success"] is True
        assert result["format"] == "text"
        assert "Dependency Graph:" in result["graph"]

    @pytest.mark.unit
    def test_nonexistent_file_typed_error(self, tmp_path: Any) -> None:
        """A missing path yields a typed failure, not an exception."""
        missing = tmp_path / "does_not_exist.md"
        result = multimodel_mcp.generate_dependency_graph_mcp(str(missing))
        assert result["success"] is False
        assert "GNN file not found" in result["error"]
        assert str(missing) in result["error"]

    @pytest.mark.unit
    def test_invalid_output_format_typed_error(self, tmp_path: Any) -> None:
        """An unsupported output_format yields the format error."""
        gnn = tmp_path / "ok.md"
        gnn.write_text("## StateSpaceBlock\n- a[2,2,type=float]\n")
        result = multimodel_mcp.generate_dependency_graph_mcp(
            str(gnn), output_format="dot"
        )
        assert result["success"] is False
        assert "output_format must be 'mermaid' or 'text'" in result["error"]


class TestRegisterTools:
    """Tests for the MCP registration surface."""

    @pytest.mark.unit
    def test_registers_single_named_tool(self) -> None:
        """Exactly one tool is registered with the expected name."""
        registry = _FakeRegistry()
        multimodel_mcp.register_tools(registry)
        names = [r["name"] for r in registry.registered]
        assert names == ["generate_dependency_graph"]

    @pytest.mark.unit
    def test_schema_requires_file_path(self) -> None:
        """The schema requires file_path and exposes the format enum."""
        registry = _FakeRegistry()
        multimodel_mcp.register_tools(registry)
        record = registry.registered[0]
        schema = record["schema"]
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"file_path"}
        props = schema["properties"]
        assert props["file_path"]["type"] == "string"
        assert props["output_format"]["enum"] == ["mermaid", "text"]
        assert props["output_format"]["default"] == "mermaid"
        assert record["category"] == "multimodel"
