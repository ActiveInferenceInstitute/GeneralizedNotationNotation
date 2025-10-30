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
        from .mcp import mcp_instance

        # Attempt to import the module's mcp.py and call its register_tools(mcp_instance)
        import importlib
        # Try both import paths: with and without 'src' prefix
        module_paths = [f"{module_name}.mcp", f"src.{module_name}.mcp"]
        module = None
        
        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                break
            except (ImportError, ModuleNotFoundError):
                continue
        
        if module is None:
            logger.warning(f"Module {module_name}.mcp not importable from paths: {module_paths}")
            return False

        if hasattr(module, "register_tools") and callable(module.register_tools):
            try:
                module.register_tools(mcp_instance)
                logger.info(f"Registered tools for module: {module_name}")
                return True
            except Exception as e:
                logger.error(f"register_tools failed for module {module_name}: {e}")
                return False
        else:
            logger.warning(f"Module src.{module_name}.mcp has no register_tools function")
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
        from datetime import datetime
        from . import __version__ as mcp_version
        
        tools = mcp_instance.list_available_tools()
        resources = mcp_instance.list_available_resources()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "mcp_version": mcp_version,
            "tools_count": len(tools),
            "resources_count": len(resources),
            "tools": tools,
            "resources": resources,
            "status": "healthy"
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate MCP report: {e}")
        import time
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }

def process_mcp(
    target_dir,
    output_dir,
    verbose: bool = False,
    logger=None,
    **kwargs
) -> bool:
    """
    Process MCP for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process (Path or str)
        output_dir: Directory to save results (Path or str)
        verbose: Enable verbose output
        logger: Optional logger instance
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    import json
    from datetime import datetime
    
    # Use provided logger or create one
    if logger is None:
        logger = logging.getLogger("mcp")
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Convert to Path objects
    target_dir = Path(target_dir) if not isinstance(target_dir, Path) else target_dir
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
    
    try:
        log_step_start(logger, "Processing MCP")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results subdirectory for detailed reports
        results_dir = output_dir / "mcp_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MCP
        from .mcp import initialize, mcp_instance
        from . import __version__ as mcp_version
        # Proceed even if SDK is missing to allow degraded mode in pipeline
        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        
        # Register tools for all modules that provide MCP adapters
        modules = [
            "gnn", "ontology", "audio", "visualization", "export", "execute", "render",
            "llm", "website", "report", "template", "validation", "setup", "model_registry",
            "integration", "security", "research", "pipeline", "tests", "type_checker", "utils"
        ]
        registered_count = 0
        registered_modules = []
        
        for module in modules:
            if register_module_tools(module):
                registered_count += 1
                registered_modules.append(module)
        
        # Get available tools after registration
        available_tools = get_available_tools()
        tools_count = len(available_tools)
        
        # Generate detailed report
        report = generate_mcp_report()
        report["registered_modules"] = registered_count
        report["registered_module_names"] = registered_modules
        report["timestamp"] = datetime.now().isoformat()
        report["target_dir"] = str(target_dir)
        report["output_dir"] = str(output_dir)
        
        # Save detailed report to subdirectory
        results_file = results_dir / "mcp_results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📋 Detailed MCP report saved to: {results_file}")
        
        # Save registered tools list
        if available_tools:
            tools_file = output_dir / "registered_tools.json"
            with open(tools_file, 'w') as f:
                json.dump(available_tools, f, indent=2)
            logger.info(f"🔧 Registered tools saved to: {tools_file}")
        
        # Create processing summary in output_dir root (for pipeline consistency)
        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "processing_status": "completed",
            "mcp_version": mcp_version,
            "tools_registered": tools_count,
            "registered_modules_count": registered_count,
            "registered_modules": registered_modules,
            "resources_count": report.get('resources_count', 0),
            "message": f"MCP processing completed - {tools_count} tools registered from {registered_count} modules"
        }
        
        summary_file = output_dir / "mcp_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ MCP summary saved to: {summary_file}")
        
        if registered_count > 0:
            log_step_success(logger, f"MCP processing completed successfully - {tools_count} tools from {registered_count} modules registered")
        else:
            log_step_warning(logger, "MCP processing completed with no modules registered")
        
        return True
        
    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {e}")
        # Still create a summary file even on error
        try:
            from . import __version__ as mcp_version
            summary = {
                "timestamp": datetime.now().isoformat(),
                "target_dir": str(target_dir),
                "output_dir": str(output_dir),
                "processing_status": "failed",
                "mcp_version": mcp_version,
                "tools_registered": 0,
                "error": str(e),
                "message": f"MCP processing failed: {str(e)}"
            }
            summary_file = output_dir / "mcp_processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except:
            pass  # If we can't even save the error summary, just log it
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