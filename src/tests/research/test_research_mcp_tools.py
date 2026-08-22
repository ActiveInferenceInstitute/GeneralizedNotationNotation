#!/usr/bin/env python3
"""
No-mock tests for the research module MCP tool handlers.

Exercises the handler functions in ``src/research/mcp.py`` directly against
real temporary directories and real GNN content. The research processor is a
rule-based static analyzer (no external LLM), so these integrations are fast
and dependency-free.

Test Coverage:
- process_research_mcp() on a real directory and a nonexistent target
- list_research_topics_mcp() taxonomy shape and required topics
- read_research_results_mcp() reading JSON + non-JSON files, directory missing
- get_research_module_info_mcp() metadata shape
- register_tools() binds the expected tool names
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research import mcp as research_mcp


class TestResearchMCPTools:
    """Functional tests for the research MCP tool handlers."""

    def _gnn_dir(self, tmp_path: Any) -> Path:
        """Create a simple GNN file with a basic state space."""
        target = tmp_path / "input"
        target.mkdir(exist_ok=True)
        (target / "simple_model.md").write_text(
            "# Simple Model\n\n"
            "## ModelName\nSimpleTest\n\n"
            "## StateSpaceBlock\n"
            "A[3,3,type=float]\n"
            "s[3,1,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.mark.unit
    def test_list_research_topics_taxonomy(self) -> Any:
        """The topic taxonomy should be non-empty and include known topics."""
        result = research_mcp.list_research_topics_mcp()
        assert result["success"] is True
        assert result["count"] == len(result["topics"]) > 0
        names = set(result["topics"].keys())
        assert "active_inference" in names
        assert "pomdp" in names
        assert "markov_blankets" in names

    @pytest.mark.unit
    def test_get_research_module_info(self) -> Any:
        """Module info should report expected capabilities and formats."""
        result = research_mcp.get_research_module_info_mcp()
        assert result["success"] is True
        assert result["module"] == "research"
        assert "cross_reference" in result["capabilities"]
        assert "json" in result["supported_output_formats"]

    @pytest.mark.unit
    def test_read_research_results_from_directory(self, tmp_path: Any) -> Any:
        """Reading a directory of JSON results should surface parsed content."""
        out = tmp_path / "results"
        out.mkdir(exist_ok=True)
        (out / "research_results.json").write_text(
            json.dumps({"processed_files": 2, "success": True})
        )
        result = research_mcp.read_research_results_mcp(str(out))
        assert result["success"] is True
        assert result["results_found"] == 1
        assert result["results"][0]["file"] == "research_results.json"
        assert result["results"][0]["content"]["processed_files"] == 2

    @pytest.mark.unit
    def test_read_research_results_non_json_content(self, tmp_path: Any) -> Any:
        """A .json file with invalid JSON should return truncated text, not crash."""
        out = tmp_path / "results"
        out.mkdir(exist_ok=True)
        (out / "results.json").write_text("plain text that is not json")
        result = research_mcp.read_research_results_mcp(str(out))
        assert result["success"] is True
        assert result["results_found"] == 1
        assert isinstance(result["results"][0]["content"], str)

    @pytest.mark.unit
    def test_read_research_results_missing_directory(self, tmp_path: Any) -> Any:
        """A missing results directory should yield a clear error."""
        missing = tmp_path / "no_results"
        result = research_mcp.read_research_results_mcp(str(missing))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.unit
    def test_process_research_mcp_on_directory(self, tmp_path: Any) -> Any:
        """A real rule-based run over a directory should report success."""
        target = self._gnn_dir(tmp_path)
        out = tmp_path / "output"
        result = research_mcp.process_research_mcp(str(target), str(out))
        assert result["success"] is True
        assert result["target_directory"] == str(target)
        assert (out / "research_results.json").exists()

    @pytest.mark.unit
    def test_process_research_mcp_nonexistent_target(self, tmp_path: Any) -> Any:
        """A nonexistent target should not raise; it returns a graceful result."""
        missing = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        result = research_mcp.process_research_mcp(str(missing), str(out))
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_register_tools_binds_research_tools(self) -> Any:
        """register_tools should bind the four research tool names."""
        registered: list[str] = []

        class StubMCP:
            def register_tool(self, name, handler, schema, description, **kw):  # noqa: ANN001
                registered.append(name)

        research_mcp.register_tools(StubMCP())  # type: ignore[arg-type]
        expected = {
            "process_research",
            "list_research_topics",
            "read_research_results",
            "get_research_module_info",
        }
        assert expected.issubset(set(registered))