"""
Model Context Protocol (MCP) Module for GNN

This module provides the Model Context Protocol implementation for the GNN project,
enabling standardized tool discovery, registration, and execution across all modules.
"""

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
__all__ = [
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
    'register_module_tools',
    'handle_mcp_request',
    'generate_mcp_report',
    'FEATURES',
    '__version__',
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
    'get_resource_info'
]


def register_module_tools(module_name: str) -> bool:
    """
    Register tools for a specific module.
    
    Args:
        module_name: Name of the module to register tools for
    
    Returns:
        True if tools registered successfully
    """
    try:
        mcp = get_mcp_instance()
        # This would typically discover and register tools for the specific module
        # For now, we'll just return success
        return True
    except Exception as e:
        import logging
        logger = logging.getLogger("mcp")
        logger.error(f"Failed to register tools for module {module_name}: {e}")
        return False


def handle_mcp_request(request: dict) -> dict:
    """
    Handle an MCP request.
    
    Args:
        request: MCP request dictionary
        
    Returns:
        MCP response dictionary
    """
    try:
        mcp = get_mcp_instance()
        return mcp.handle_request(request)
    except Exception as e:
        import logging
        logger = logging.getLogger("mcp")
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
    Generate a comprehensive MCP report.
    
    Returns:
        Dictionary containing MCP status and metrics
    """
    try:
        mcp = get_mcp_instance()
        return mcp.get_enhanced_server_status()
    except Exception as e:
        import logging
        logger = logging.getLogger("mcp")
        logger.error(f"Failed to generate MCP report: {e}")
        return {
            "error": f"Failed to generate report: {str(e)}",
            "timestamp": __import__('time').time()
        }


def get_available_tools() -> list:
    """
    Get list of all available tools.
    
    Returns:
        List of tool names
    """
    try:
        return list_available_tools()
    except Exception as e:
        import logging
        logger = logging.getLogger("mcp")
        logger.error(f"Failed to get available tools: {e}")
        return [] 