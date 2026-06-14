"""
GUI MCP Integration

This module exposes GUI processing tools via MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import utilities from the gui module
from . import FEATURES, get_available_guis, process_gui

# MCP Tools for GUI Module


def process_gui_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    gui_types: str = "gui_1,gui_2",
    headless: bool = True,
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
            headless=headless,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "gui_types": gui_types.split(","),
            "mode": "headless" if headless else "interactive",
            "message": f"GUI processing {'completed successfully' if success else 'failed'}",
        }
    except Exception as e:
        logger.error(
            f"Error in process_gui_mcp for {target_directory}: {e}", exc_info=True
        )
        return {"success": False, "error": str(e)}


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
            "success": True,
            "available_guis": guis,
            "features": FEATURES,
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

        mod = importlib.import_module(__package__)
        version = getattr(mod, "__version__", "unknown")
        return {
            "success": True,
            "module": __package__,
            "version": version,
            "features": FEATURES,
            "gui_types": ["gui_1", "gui_2", "gui_3", "oxdraw"],
            "tools": [
                "process_gui",
                "list_available_guis",
                "get_gui_module_info",
                "oxdraw.convert_to_mermaid",
                "oxdraw.convert_from_mermaid",
                "oxdraw.launch_editor",
                "oxdraw.check_installation",
                "oxdraw.get_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_gui_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def oxdraw_convert_to_mermaid_mcp(
    gnn_file_path: str,
    output_path: Optional[str] = None,
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Convert a GNN file to Mermaid format through the oxdraw submodule."""
    from .oxdraw.mcp import tool_convert_to_mermaid

    args: Dict[str, Any] = {
        "gnn_file_path": gnn_file_path,
        "include_metadata": include_metadata,
    }
    if output_path is not None:
        args["output_path"] = output_path
    return tool_convert_to_mermaid(args)


def oxdraw_convert_from_mermaid_mcp(
    mermaid_file_path: str,
    output_path: Optional[str] = None,
    validate_ontology: bool = False,
) -> Dict[str, Any]:
    """Convert a Mermaid file back to GNN format through the oxdraw submodule."""
    from .oxdraw.mcp import tool_convert_from_mermaid

    args: Dict[str, Any] = {
        "mermaid_file_path": mermaid_file_path,
        "validate_ontology": validate_ontology,
    }
    if output_path is not None:
        args["output_path"] = output_path
    return tool_convert_from_mermaid(args)


def oxdraw_launch_editor_mcp(
    mermaid_file_path: str,
    port: int = 5151,
    host: str = "127.0.0.1",
) -> Dict[str, Any]:
    """Launch the oxdraw editor for a Mermaid file."""
    from .oxdraw.mcp import tool_launch_editor

    return tool_launch_editor(
        {"mermaid_file_path": mermaid_file_path, "port": port, "host": host}
    )


def oxdraw_check_installation_mcp() -> Dict[str, Any]:
    """Return oxdraw CLI installation status."""
    from .oxdraw.mcp import tool_check_installation

    return tool_check_installation({})


def oxdraw_get_info_mcp() -> Dict[str, Any]:
    """Return oxdraw integration metadata."""
    from .oxdraw.mcp import tool_get_info

    return tool_get_info({})


# MCP Registration Function
def register_tools(mcp_instance: Any) -> Any:
    """Register GUI domain tools with the MCP."""

    mcp_instance.register_tool(
        "process_gui",
        process_gui_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory containing GNN files to process",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save GUI outputs",
                },
                "verbose": {"type": "boolean", "default": False},
                "gui_types": {
                    "type": "string",
                    "default": "gui_1,gui_2",
                    "description": "Comma-separated GUI types",
                },
                "headless": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run in headless mode",
                },
            },
            "required": ["target_directory", "output_directory"],
        },
        "Process GUI generation for GNN files: form editors, visual constructors, OxDraw diagram tools.",
        module=__package__,
        category="gui",
    )

    mcp_instance.register_tool(
        "list_available_guis",
        list_available_guis_mcp,
        {},
        "List all available GUI implementations (gui_1, gui_2, gui_3, oxdraw) with capabilities.",
        module=__package__,
        category="gui",
    )

    mcp_instance.register_tool(
        "get_gui_module_info",
        get_gui_module_info_mcp,
        {},
        "Return version, feature flags, GUI types, and tool inventory of the GUI module.",
        module=__package__,
        category="gui",
    )

    oxdraw_tools = {
        "oxdraw.convert_to_mermaid": oxdraw_convert_to_mermaid_mcp,
        "oxdraw.convert_from_mermaid": oxdraw_convert_from_mermaid_mcp,
        "oxdraw.launch_editor": oxdraw_launch_editor_mcp,
        "oxdraw.check_installation": oxdraw_check_installation_mcp,
        "oxdraw.get_info": oxdraw_get_info_mcp,
    }
    try:
        from .oxdraw.mcp import register_mcp_tools

        for tool in register_mcp_tools():
            name = tool["name"]
            mcp_instance.register_tool(
                name,
                oxdraw_tools[name],
                tool.get("input_schema", {}),
                tool.get("description", ""),
                module="src.gui.oxdraw",
                category="gui",
            )
    except Exception as e:
        logger.error("Failed to register oxdraw MCP tools: %s", e, exc_info=True)
        raise

    logger.info("gui module MCP tools registered (8 domain tools).")
