"""
MCP (Model Context Protocol) integration for website utilities.

This module exposes website rendering functions through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from .renderer import process_website

import inspect, importlib

def _get_module_pkg():
    try:
        return importlib.import_module(__package__)
    except Exception:
        import sys
        return sys.modules.get(__package__)

def list_functions_mcp() -> Dict[str, Any]:
    module_pkg = _get_module_pkg()
    public_names = getattr(module_pkg, "__all__", []) or [n for n in dir(module_pkg) if not n.startswith("_")]
    funcs = []
    for name in public_names:
        obj = getattr(module_pkg, name, None)
        if inspect.isfunction(obj):
            funcs.append(name)
    return {"success": True, "module": __package__, "functions": sorted(set(funcs))}

def call_function_mcp(function_name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    from pathlib import Path
    module_pkg = _get_module_pkg()
    func = getattr(module_pkg, function_name, None)
    if not callable(func):
        return {"success": False, "error": f"Function not found or not callable: {function_name}"}
    arguments = arguments or {}
    converted: Dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and any(token in key.lower() for token in ["dir", "path", "file", "output", "input"]):
            converted[key] = Path(value)
        else:
            converted[key] = value
    try:
        result = func(**converted)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"success": False, "error": f"TypeError: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# MCP Tools for $module Utilities Module

def process_website_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Process website generation for pipeline artifacts. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_website(Path(target_directory), Path(output_directory), verbose=verbose)
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"website processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_$module_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register website utility tools with the MCP."""
    
    # Generic namespaced tools
    mcp_instance.register_tool(
        f"{__package__}.list_functions",
        list_functions_mcp,
        {},
        f"List callable functions exported by the {__package__} module public API."
    )
    mcp_instance.register_tool(
        f"{__package__}.call_function",
        call_function_mcp,
        {
            "function_name": {"type": "string", "description": "Function name exported by the module"},
            "arguments": {"type": "object", "description": "Keyword arguments for the function", "default": {}}
        },
        f"Call any public function in the {__package__} module with keyword arguments."
    )

    mcp_instance.register_tool(
        "process_website",
        process_website_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save website results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Generate website from pipeline artifacts in the specified directory."
    )
    
    logger.info("website module MCP tools registered.")
