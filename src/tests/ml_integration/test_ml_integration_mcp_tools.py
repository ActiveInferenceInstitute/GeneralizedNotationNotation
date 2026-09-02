#!/usr/bin/env python3
"""
Exercises the ML Integration module MCP tool handlers.

Exercises the handler functions in ``src/ml_integration/mcp.py`` directly.
Framework probe handlers use metadata-only lookups (importlib.find_spec), so
they run fast and without heavy framework imports. The slow training path
(``process_ml_integration``) is only exercised via its graceful failure mode.

Test Coverage:
- check_ml_frameworks_mcp() returns a frameworks map with available/version keys
- list_ml_integration_targets_mcp() target inventory shape
- get_ml_module_info_mcp() version/features/tools shape
- register_tools() binds the expected tool names
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_integration import mcp as ml_mcp


class TestMLIntegrationMCPTools:
    """Functional tests for the ML integration MCP tool handlers."""

    @pytest.mark.unit
    def test_check_ml_frameworks_reports_availability(self) -> Any:
        """Framework check should report an availability flag per framework."""
        result = ml_mcp.check_ml_frameworks_mcp()
        assert result["success"] is True
        assert isinstance(result["frameworks"], dict)
        assert "features" in result
        # Known probed frameworks must be present regardless of env availability.
        for fw in ("pytorch", "tensorflow", "jax", "sklearn"):
            assert fw in result["frameworks"]
            assert "available" in result["frameworks"][fw]

    @pytest.mark.unit
    def test_list_ml_integration_targets_shape(self) -> Any:
        """Target inventory should list downstream consumers with booleans."""
        result = ml_mcp.list_ml_integration_targets_mcp()
        assert result["success"] is True
        assert "pymdp" in result["targets"]
        assert "jax" in result["targets"]
        assert "torch" in result["targets"]
        assert all(isinstance(v, bool) for v in result["targets"].values())
        assert result["count"] == len(result["available"])
        assert set(result["available"]).issubset(result["targets"].keys())

    @pytest.mark.unit
    def test_get_ml_module_info_metadata(self) -> Any:
        """Module info should include version, features, and the tool list."""
        result = ml_mcp.get_ml_module_info_mcp()
        assert result["success"] is True
        assert result["module"] == "ml_integration"
        assert "version" in result and result["version"]
        assert "process_ml_integration" in result["tools"]
        assert "check_ml_frameworks" in result["tools"]

    @pytest.mark.unit
    def test_register_tools_binds_ml_tools(self) -> None:
        """register_tools should bind the four ML integration tool names."""
        registered: list[str] = []

        class StubMCP:
            def register_tool(
                self, name: str, handler: Any, schema: Any, description: str, **kw: Any
            ) -> None:
                registered.append(name)

        ml_mcp.register_tools(StubMCP())
        expected = {
            "process_ml_integration",
            "check_ml_frameworks",
            "list_ml_integration_targets",
            "get_ml_module_info",
        }
        assert expected.issubset(set(registered))
