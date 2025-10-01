"""
Model Context Protocol (MCP) Module for GNN

This module provides the Model Context Protocol implementation for the GNN project,
enabling standardized tool discovery, registration, and execution across all modules.
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

from .mcp import (
    mcp_instance,
    initialize,
    MCP,
    MCPTool,
    MCPResource,
    MCPError,
    MCPServer,
    create_mcp_server,
    start_mcp_server,
    register_tools,
    get_mcp_instance,
    # Enhanced error classes
    MCPToolNotFoundError,
    MCPResourceNotFoundError,
    MCPInvalidParamsError,
    MCPToolExecutionError,
    MCPSDKNotFoundError,
    MCPValidationError,
    MCPModuleLoadError,
    MCPPerformanceError,
    # Enhanced data structures
    MCPModuleInfo,
    MCPPerformanceMetrics,
    MCPSDKStatus,
    # Enhanced utility functions
    list_available_tools,
    list_available_resources,
    get_tool_info,
    get_resource_info
)

# Import processor functions
from .processor import (
    register_module_tools,
    handle_mcp_request,
    generate_mcp_report,
    process_mcp,
    get_available_tools
)

# Module metadata
__version__ = "2.0.0"
__author__ = "Active Inference Institute"
__description__ = "Enhanced Model Context Protocol implementation for GNN"

# Feature availability flags
FEATURES = {
    'tool_registration': True,
    'resource_access': True,
    'module_discovery': True,
    'json_rpc': True,
    'server_implementation': True,
    'error_handling': True,
    'mcp_integration': True,
    'enhanced_features': True,
    'caching': True,
    'rate_limiting': True,
    'concurrent_control': True,
    'performance_monitoring': True,
    'thread_safety': True,
    'enhanced_validation': True,
    'health_monitoring': True
}

# Main API functions

def process_mcp(target_dir, output_dir, verbose=False, logger=None, **kwargs):
    """
    Main processing function for mcp.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        logger: Logger instance
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    import json
    from pathlib import Path
    from datetime import datetime
    
    if logger is None:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing mcp for files in {target_dir}")
        
        # Get available MCP tools
        available_tools = get_available_tools() if 'get_available_tools' in globals() else []
        
        # Create processing summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "processing_status": "completed",
            "mcp_version": __version__,
            "tools_registered": len(available_tools),
            "message": "MCP module ready for tool registration and execution"
        }
        
        # Save summary
        summary_file = output_dir / "mcp_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"🔧 MCP summary saved to: {summary_file}")
        
        # Save registered tools
        if available_tools:
            tools_file = output_dir / "registered_tools.json"
            with open(tools_file, 'w') as f:
                json.dump(available_tools, f, indent=2)
            logger.info(f"📋 Registered tools saved to: {tools_file}")
        
        logger.info(f"✅ MCP processing completed")
        return True
    except Exception as e:
        logger.error(f"❌ MCP processing failed: {e}")
        return False


__all__ = [
    # Core MCP classes and functions
    'mcp_instance',
    'initialize',
    'MCP',
    'MCPTool',
    'MCPResource',
    'MCPError',
    'MCPServer',
    'create_mcp_server',
    'start_mcp_server',
    'register_tools',
    'get_mcp_instance',
    
    # Processor functions
    'register_module_tools',
    'handle_mcp_request',
    'generate_mcp_report',
    'process_mcp',
    'get_available_tools',
    
    # Enhanced error classes
    'MCPToolNotFoundError',
    'MCPResourceNotFoundError',
    'MCPInvalidParamsError',
    'MCPToolExecutionError',
    'MCPSDKNotFoundError',
    'MCPValidationError',
    'MCPModuleLoadError',
    'MCPPerformanceError',
    
    # Enhanced data structures
    'MCPModuleInfo',
    'MCPPerformanceMetrics',
    'MCPSDKStatus',
    
    # Enhanced utility functions
    'list_available_tools',
    'list_available_resources',
    'get_tool_info',
    'get_resource_info',
    
    # Metadata
    'FEATURES',
    '__version__'
] 