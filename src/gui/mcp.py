"""
GUI MCP Integration

This module exposes GUI processing tools via MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the gui module
from . import process_gui, get_available_guis, FEATURES

# MCP Tools for GUI Module


def process_gui_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    gui_types: str = "gui_1,gui_2",
    headless: bool = True
) -> Dict[str, Any]:
    """
    Process GUI generation for GNN files. Exposed via MCP.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save GUI outputs
        verbose: Enable verbose output
        gui_types: Comma-separated list of GUI types to run (gui_1, gui_2, gui_3, oxdraw)
        headless: Run in headless mode (no interactive servers)

    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_gui(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
            gui_types=gui_types,
            headless=headless
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "gui_types": gui_types.split(","),
            "mode": "headless" if headless else "interactive",
            "message": f"GUI processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_gui_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def list_available_guis_mcp() -> Dict[str, Any]:
    """
    List available GUI implementations. Exposed via MCP.

    Returns:
        Dictionary with available GUI types and their information.
    """
    try:
        guis = get_available_guis()
        return {
            "success": True,
            "available_guis": guis,
            "features": FEATURES
        }
    except Exception as e:
        logger.error(f"Error listing available GUIs: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# MCP Registration Function
def register_tools(mcp_instance):
    """Register GUI utility tools with the MCP."""

    mcp_instance.register_tool(
        "process_gui",
        process_gui_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save GUI outputs."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True},
            "gui_types": {"type": "string", "description": "Comma-separated GUI types (gui_1,gui_2,gui_3,oxdraw). Defaults to gui_1,gui_2.", "optional": True},
            "headless": {"type": "boolean", "description": "Run in headless mode. Defaults to true.", "optional": True}
        },
        "Process GUI generation for GNN files including form-based constructors, visual editors, and diagram tools."
    )

    mcp_instance.register_tool(
        "list_available_guis",
        list_available_guis_mcp,
        {},
        "List all available GUI implementations with their capabilities."
    )

    logger.info("GUI module MCP tools registered.")
