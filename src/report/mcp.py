"""
MCP (Model Context Protocol) integration for report utilities.

This module exposes report generation functions through MCP.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from .generator import generate_comprehensive_report

# MCP Tools for $module Utilities Module

def generate_report_mcp(target_directory: str, output_directory: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate comprehensive report for pipeline artifacts. Exposed via MCP.
    
    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        Dictionary with operation status and results.
    """
    try:
        # Use generator API; mimic success boolean return
        import logging as _logging
        logger = _logging.getLogger("report.mcp")
        success = generate_comprehensive_report(Path(target_directory), Path(output_directory), logger)
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"report generation {'completed successfully' if success else 'failed'}"
        }
    except Exception as e:
        logger.error(f"Error in process_$module_mcp for {target_directory}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

# MCP Registration Function
def register_tools(mcp_instance):
    """Register report utility tools with the MCP."""
    
    mcp_instance.register_tool(
        "generate_report",
        generate_report_mcp,
        {
            "target_directory": {"type": "string", "description": "Directory containing GNN files to process."},
            "output_directory": {"type": "string", "description": "Directory to save report results."},
            "verbose": {"type": "boolean", "description": "Enable verbose output. Defaults to false.", "optional": True}
        },
        "Generate comprehensive report from pipeline artifacts."
    )
    
    logger.info("report module MCP tools registered.")
