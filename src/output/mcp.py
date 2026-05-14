"""MCP tools for output fixture inspection."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_tools(mcp_instance: Any) -> None:
    """Register output fixture inspection tools with the MCP server."""

    def describe_output_fixtures() -> dict[str, Any]:
        return {
            "module": "output",
            "purpose": "Reference pipeline artifact fixtures used by integration tests",
            "live_output_dir": "output/",
            "fixture_dir": "src/output/",
        }

    mcp_instance.register_tool(
        name="output.describe_fixtures",
        func=describe_output_fixtures,
        schema={},
        description="Describe the source-controlled output fixture directory.",
    )
    logger.info("output.mcp: registered fixture description tool")
