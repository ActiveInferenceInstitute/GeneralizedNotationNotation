#!/usr/bin/env python3
"""
Exercises the integration module MCP tool handlers.

Exercises the handler functions in ``src/integration/mcp.py``: supported
integration inventory, dependency probing (metadata-only find_spec), and
output status inventory. These are dependency-light and deterministic.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integration import mcp as integration_mcp


class TestIntegrationMCPTools:
    """Tests for the integration MCP tool handlers."""

    @pytest.mark.unit
    def test_list_supported_integrations(self) -> None:
        """The integration inventory should include the known backends."""
        result = integration_mcp.list_supported_integrations_mcp()
        assert result["success"] is True
        assert result["count"] == len(result["integrations"])
        names = set(result["integrations"].keys())
        assert {"activeinference_jl", "pymdp", "rxinfer"}.issubset(names)
        for meta in result["integrations"].values():
            assert "description" in meta
            assert "output_format" in meta

    @pytest.mark.unit
    def test_check_integration_dependencies(self) -> None:
        """Dependency probe should report availability per package."""
        result = integration_mcp.check_integration_dependencies_mcp()
        assert result["success"] is True
        assert "pymdp" in result["dependencies"]
        assert "JAX" in result["dependencies"]
        assert "Julia" in result["dependencies"]
        for meta in result["dependencies"].values():
            assert "available" in meta

    @pytest.mark.unit
    def test_get_integration_status_directory(self, tmp_path: Any) -> None:
        """Status should count files by extension in an output directory."""
        out = tmp_path / "integration_out"
        out.mkdir(exist_ok=True)
        (out / "model.jl").write_text("julia")
        (out / "model.py").write_text("python")
        (out / "meta.json").write_text("{}")
        result = integration_mcp.get_integration_status_mcp(str(out))
        assert result["success"] is True
        assert result["total_files"] == 3
        assert result["by_extension"] == {"jl": 1, "py": 1, "json": 1}

    @pytest.mark.unit
    def test_get_integration_status_missing_directory(self, tmp_path: Any) -> None:
        """A missing output directory should yield a clear error."""
        missing = tmp_path / "no_out"
        result = integration_mcp.get_integration_status_mcp(str(missing))
        assert result["success"] is False
        assert "not found" in result["error"].lower()