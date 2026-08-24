"""Live registration and execution coverage for integration MCP tools."""

from __future__ import annotations

from pathlib import Path

from integration.mcp import register_tools
from mcp.mcp import MCP


def test_documented_integration_tools_register_and_execute(tmp_path: Path) -> None:
    target_dir = tmp_path / "input"
    target_dir.mkdir()
    output_dir = tmp_path / "output"

    mcp = MCP(
        enable_caching=False,
        enable_rate_limiting=False,
        strict_validation=True,
    )
    try:
        register_tools(mcp)
        assert set(mcp.tools) == {
            "process_integration",
            "list_supported_integrations",
            "get_integration_status",
            "check_integration_dependencies",
        }

        supported = mcp.execute_tool("list_supported_integrations", {})
        dependencies = mcp.execute_tool("check_integration_dependencies", {})
        processed = mcp.execute_tool(
            "process_integration",
            {
                "target_directory": str(target_dir),
                "output_directory": str(output_dir),
            },
        )
        status = mcp.execute_tool(
            "get_integration_status",
            {"output_directory": str(output_dir)},
        )
        missing = mcp.execute_tool(
            "get_integration_status",
            {"output_directory": str(tmp_path / "missing")},
        )

        assert supported["success"] is True
        assert dependencies["success"] is True
        assert processed["success"] is True
        assert status["success"] is True
        assert missing["success"] is False
        assert "not found" in missing["error"].lower()
    finally:
        mcp.shutdown()
