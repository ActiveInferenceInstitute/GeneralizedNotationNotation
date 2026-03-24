"""
GUI MCP Integration

This module exposes GUI processing tools via MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Import utilities from the gui module
from . import FEATURES, get_available_guis, process_gui

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
    List all available GUI implementations and their capabilities.

    Returns the list of supported GUI types (gui_1, gui_2, gui_3, oxdraw),
    their display names, dependency requirements, and feature flags.

    Returns:
        Dictionary with GUI types, capabilities, and feature inventory.
    """
    try:
        guis = get_available_guis()
        return {
            "success":        True,
            "available_guis": guis,
            "features":       FEATURES,
        }
    except Exception as e:
        logger.error(f"list_available_guis_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_gui_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the GUI module.

    Returns:
        Dictionary with module metadata, GUI types, and tool inventory.
    """
    try:
        import importlib
        mod     = importlib.import_module(__package__)
        version = getattr(mod, "__version__", "unknown")
        return {
            "success":   True,
            "module":    __package__,
            "version":   version,
            "features":  FEATURES,
            "gui_types": ["gui_1", "gui_2", "gui_3", "oxdraw"],
            "tools": [
                "process_gui",
                "list_available_guis",
                "get_gui_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_gui_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}



# MCP Registration Function
def register_tools(mcp_instance):
    """Register GUI domain tools with the MCP."""

    mcp_instance.register_tool(
        "process_gui",
        process_gui_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN files to process"},
                "output_directory": {"type": "string", "description": "Directory to save GUI outputs"},
                "verbose":          {"type": "boolean", "default": False},
                "gui_types":        {"type": "string", "default": "gui_1,gui_2", "description": "Comma-separated GUI types"},
                "headless":         {"type": "boolean", "default": True, "description": "Run in headless mode"},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Process GUI generation for GNN files: form editors, visual constructors, OxDraw diagram tools.",
        module=__package__, category="gui",
    )

    mcp_instance.register_tool(
        "list_available_guis",
        list_available_guis_mcp,
        {},
        "List all available GUI implementations (gui_1, gui_2, gui_3, oxdraw) with capabilities.",
        module=__package__, category="gui",
    )

    mcp_instance.register_tool(
        "get_gui_module_info",
        get_gui_module_info_mcp,
        {},
        "Return version, feature flags, GUI types, and tool inventory of the GUI module.",
        module=__package__, category="gui",
    )

    logger.info("gui module MCP tools registered (3 domain tools).")

