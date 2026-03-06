"""
Intelligent Analysis MCP Integration

This module exposes intelligent analysis processing tools via MCP.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Import utilities from the intelligent_analysis module
from . import (
    process_intelligent_analysis,
    get_module_info,
    get_supported_analysis_types,
    check_analysis_tools,
    FEATURES
)

# MCP Tools for Intelligent Analysis Module


def process_intelligent_analysis_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    analysis_types: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process intelligent analysis for pipeline results. Exposed via MCP.

    Args:
        target_directory: Directory containing pipeline results to analyze
        output_directory: Directory to save analysis results
        verbose: Enable verbose output
        analysis_types: Comma-separated list of analysis types (optional)

    Returns:
        Dictionary with operation status and results.
    """
    try:
        kwargs = {}
        if analysis_types:
            kwargs['analysis_types'] = analysis_types.split(',')

        success = process_intelligent_analysis(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
            **kwargs
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Intelligent analysis {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_intelligent_analysis_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def get_analysis_capabilities_mcp() -> Dict[str, Any]:
    """
    Get intelligent analysis capabilities and available tools.

    Returns the full module info, all supported analysis types, tool
    availability status, and active feature flags.

    Returns:
        Dictionary with capabilities, analysis types, and feature inventory.
    """
    try:
        return {
            "success":                True,
            "module_info":            get_module_info(),
            "supported_analysis_types": get_supported_analysis_types(),
            "available_tools":        check_analysis_tools(),
            "features":               FEATURES,
        }
    except Exception as e:
        logger.error(f"get_analysis_capabilities_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_intelligent_analysis_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the intelligent analysis module.

    Returns:
        Dictionary with module metadata, supported analysis types, and tool list.
    """
    try:
        import importlib
        mod     = importlib.import_module(__package__)
        version = getattr(mod, "__version__", "unknown")
        return {
            "success":                True,
            "module":                 __package__,
            "version":                version,
            "features":               FEATURES,
            "supported_analysis_types": get_supported_analysis_types(),
            "tools": [
                "process_intelligent_analysis",
                "get_analysis_capabilities",
                "get_intelligent_analysis_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_intelligent_analysis_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}



# MCP Registration Function
def register_tools(mcp_instance):
    """Register intelligent analysis domain tools with the MCP."""

    mcp_instance.register_tool(
        "process_intelligent_analysis",
        process_intelligent_analysis_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing pipeline results to analyze"},
                "output_directory": {"type": "string", "description": "Directory to save analysis results"},
                "verbose":          {"type": "boolean", "default": False},
                "analysis_types":   {"type": "string", "description": "Comma-separated analysis types (optional)", "nullable": True},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Process AI-powered intelligent analysis of GNN pipeline results: failure analysis, performance optimization.",
        module=__package__, category="intelligent_analysis",
    )

    mcp_instance.register_tool(
        "get_analysis_capabilities",
        get_analysis_capabilities_mcp,
        {},
        "Get intelligent analysis capabilities, supported types, available tools, and feature flags.",
        module=__package__, category="intelligent_analysis",
    )

    mcp_instance.register_tool(
        "get_intelligent_analysis_module_info",
        get_intelligent_analysis_module_info_mcp,
        {},
        "Return version, feature flags, and tool inventory of the intelligent analysis module.",
        module=__package__, category="intelligent_analysis",
    )

    logger.info("intelligent_analysis module MCP tools registered (3 domain tools).")

