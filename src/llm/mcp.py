"""
MCP (Model Context Protocol) integration for llm utilities.

This module exposes utility functions from the llm module through MCP.
"""

import importlib
import inspect
import logging
from pathlib import Path
from . import process_llm

logger = logging.getLogger(__name__)

def _get_module_pkg():
    try:
        return importlib.import_module(__package__)
    except Exception:
        import sys
        return sys.modules.get(__package__)

def list_functions_mcp() -> dict:
    module_pkg = _get_module_pkg()
    public_names = getattr(module_pkg, "__all__", []) or [n for n in dir(module_pkg) if not n.startswith("_")]
    funcs = []
    for name in public_names:
        obj = getattr(module_pkg, name, None)
        if inspect.isfunction(obj):
            funcs.append(name)
    return {"success": True, "module": __package__, "functions": sorted(set(funcs))}

def call_function_mcp(function_name: str, arguments: dict | None = None) -> dict:
    from pathlib import Path
    module_pkg = _get_module_pkg()
    func = getattr(module_pkg, function_name, None)
    if not callable(func):
        return {"success": False, "error": f"Function not found or not callable: {function_name}"}
    arguments = arguments or {}
    converted: dict = {}
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
"""MCP Tools for LLM Utilities Module"""
def process_llm_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> dict:
    """
    Process llm for GNN files. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        success = process_llm(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"llm processing {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_llm_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register llm utility tools with the MCP."""
    
    # Generic namespaced tools
    mcp_instance.register_tool(
        f"{__package__}.list_functions",
        list_functions_mcp,
        {},
        "List callable functions exported by the module public API."
    )
    mcp_instance.register_tool(
        f"{__package__}.call_function",
        call_function_mcp,
        {
            "function_name": {"type": "string", "description": "Function name exported by the module"},
            "arguments": {"type": "object", "description": "Keyword arguments for the function", "default": {}}
        },
        "Call any public function in the module with keyword arguments."
    )

    mcp_instance.register_tool(
        "process_llm",
        process_llm_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save llm results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Process llm for GNN files in the specified directory."
    )
    logger.info("llm module MCP tools registered.")
