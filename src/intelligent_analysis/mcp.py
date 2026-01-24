"""
Intelligent Analysis MCP Integration

This module exposes intelligent analysis processing tools via MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    Get intelligent analysis capabilities. Exposed via MCP.

    Returns:
        Dictionary with available analysis features and tools.
    """
    try:
        return {
            "success": True,
            "module_info": get_module_info(),
            "supported_analysis_types": get_supported_analysis_types(),
            "available_tools": check_analysis_tools(),
            "features": FEATURES
        }
    except Exception as e:
        logger.error(f"Error getting analysis capabilities: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# MCP Registration Function
def register_tools(mcp_instance):
    """Register intelligent analysis utility tools with the MCP."""

    mcp_instance.register_tool(
        "process_intelligent_analysis",
        process_intelligent_analysis_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing pipeline results to analyze."},
            "output_directory": {"type": "string", "description": "Directory to save analysis results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True},
            "analysis_types": {"type": "string", "description": "Comma-separated analysis types to run. Defaults to all.", "optional": True}
        },
        "Process intelligent AI-powered analysis of pipeline results including failure analysis and performance optimization."
    )

    mcp_instance.register_tool(
        "get_analysis_capabilities",
        get_analysis_capabilities_mcp,
        {},
        "Get available intelligent analysis capabilities, tools, and features."
    )

    logger.info("Intelligent analysis module MCP tools registered.")
