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
__version__ = "1.1.3"
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
# Note: process_mcp is imported from processor.py above, not redefined here


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