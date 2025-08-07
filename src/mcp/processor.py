#!/usr/bin/env python3
"""
MCP Processor module for GNN Processing Pipeline.

This module provides MCP processing capabilities.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

logger = logging.getLogger(__name__)

def register_module_tools(module_name: str) -> bool:
    """
    Register tools for a specific module.
    
    Args:
        module_name: Name of the module to register tools for
        
    Returns:
        True if registration successful, False otherwise
    """
    try:
        logger.info(f"Registering tools for module: {module_name}")
        
        # Import MCP instance
        from .mcp import mcp_instance, MCPTool
        
        # Define module-specific tools
        module_tools = {
            "gnn": [
                MCPTool("parse_gnn", "Parse GNN file", "gnn_parser"),
                MCPTool("validate_gnn", "Validate GNN syntax", "gnn_validator"),
                MCPTool("convert_gnn", "Convert GNN format", "gnn_converter")
            ],
            "ontology": [
                MCPTool("process_ontology", "Process ontology", "ontology_processor"),
                MCPTool("validate_ontology", "Validate ontology", "ontology_validator")
            ],
            "audio": [
                MCPTool("generate_audio", "Generate audio from GNN", "audio_generator"),
                MCPTool("process_sapf", "Process SAPF audio", "sapf_processor")
            ],
            "visualization": [
                MCPTool("create_visualization", "Create visualization", "viz_creator"),
                MCPTool("generate_plot", "Generate plot", "plot_generator")
            ]
        }
        
        if module_name in module_tools:
            tools = module_tools[module_name]
            for tool in tools:
                mcp_instance.register_tool(tool)
                logger.debug(f"Registered tool: {tool.name}")
            
            logger.info(f"Successfully registered {len(tools)} tools for {module_name}")
            return True
        else:
            logger.warning(f"No tools defined for module: {module_name}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to register tools for {module_name}: {e}")
        return False

def handle_mcp_request(request: dict) -> dict:
    """
    Handle MCP request.
    
    Args:
        request: MCP request dictionary
        
    Returns:
        Response dictionary
    """
    try:
        logger.info(f"Handling MCP request: {request.get('method', 'unknown')}")
        
        # Import MCP instance
        from .mcp import mcp_instance
        
        # Process request
        method = request.get("method", "")
        params = request.get("params", {})
        
        if method == "tools/list":
            tools = mcp_instance.list_available_tools()
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"tools": tools}
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_params = params.get("arguments", {})
            
            result = mcp_instance.execute_tool(tool_name, tool_params)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to handle MCP request: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

def generate_mcp_report() -> dict:
    """
    Generate MCP report.
    
    Returns:
        Dictionary with MCP report
    """
    try:
        from .mcp import mcp_instance
        
        tools = mcp_instance.list_available_tools()
        resources = mcp_instance.list_available_resources()
        
        report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "mcp_version": "2.0.0",
            "tools_count": len(tools),
            "resources_count": len(resources),
            "tools": tools,
            "resources": resources,
            "status": "healthy"
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate MCP report: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

def process_mcp(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process MCP for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("mcp")
    
    try:
        log_step_start(logger, "Processing MCP")
        
        # Create results directory
        results_dir = output_dir / "mcp_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MCP
        from .mcp import initialize
        initialize()
        
        # Register tools for all modules
        modules = ["gnn", "ontology", "audio", "visualization"]
        registered_count = 0
        
        for module in modules:
            if register_module_tools(module):
                registered_count += 1
        
        # Generate report
        report = generate_mcp_report()
        report["registered_modules"] = registered_count
        
        # Save results
        import json
        results_file = results_dir / "mcp_results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if registered_count > 0:
            log_step_success(logger, f"MCP processing completed successfully - {registered_count} modules registered")
        else:
            log_step_warning(logger, "MCP processing completed with no modules registered")
        
        return True
        
    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {e}")
        return False

def get_available_tools() -> list:
    """
    Get list of available MCP tools.
    
    Returns:
        List of available tools
    """
    try:
        from .mcp import mcp_instance
        return mcp_instance.list_available_tools()
    except Exception as e:
        logger.error(f"Failed to get available tools: {e}")
        return [] 