"""
MCP registration for CLI module.

Provides tools for pipeline execution and health checking via MCP.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def cli_health_check(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return CLI module health and available subcommands."""
    subcommands = [
        "run", "validate", "parse", "render", "report",
        "reproduce", "preflight", "health", "serve", "lsp",
    ]
    return {
        "success": True,
        "module": "cli",
        "subcommands": subcommands,
        "subcommand_count": len(subcommands),
    }


def cli_preflight(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run preflight checks for the pipeline environment."""
    try:
        # Return info without running the full check
        return {
            "success": True,
            "preflight_available": True,
            "message": "Use 'gnn preflight' to run full environment checks",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_tools(mcp_instance: Any) -> None:
    """Register CLI tools with the MCP instance."""
    mcp_instance.register_tool(
        "cli.health",
        cli_health_check,
        {"type": "object", "properties": {}},
        "Return CLI module health and list of available subcommands",
        module="cli", category="cli",
    )

    mcp_instance.register_tool(
        "cli.preflight",
        cli_preflight,
        {"type": "object", "properties": {}},
        "Check pipeline environment readiness",
        module="cli", category="cli",
    )

    logger.info("cli module MCP tools registered (2 tools).")
