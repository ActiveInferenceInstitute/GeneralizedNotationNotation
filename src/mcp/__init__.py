"""MCP (Model Context Protocol) integration for the GNN pipeline: tool discovery, registration, and execution.

See ``src/mcp/AGENTS.md`` for the public API, step-21 wiring, and tool registration patterns.
"""

# Import exception classes from exceptions module
from .exceptions import (
    MCPError,
    MCPInvalidParamsError,
    MCPModuleLoadError,
    MCPPerformanceError,
    MCPResourceNotFoundError,
    MCPSDKNotFoundError,
    MCPToolExecutionError,
    MCPToolNotFoundError,
    MCPValidationError,
)
from .mcp import (
    MCP,
    get_mcp_instance,
    get_resource_info,
    get_tool_info,
    initialize,
    list_available_resources,
    # Enhanced utility functions
    list_available_tools,
    mcp_instance,
)

# Import data structures from models
from .models import (
    MCPModuleInfo,
    MCPPerformanceMetrics,
    MCPResource,
    MCPSDKStatus,
    MCPTool,
)

# Import processor functions
from .processor import (
    generate_mcp_report,
    get_available_tools,
    handle_mcp_request,
    process_mcp,
    register_module_tools,
)
from .server import MCPServer
from .server_core import create_mcp_server, start_mcp_server

# Backward-compatible alias expected by legacy tests/import sites.
MCPServer = MCP

# Module metadata
__version__ = "1.6.0"
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
    'MCPServer',
    'MCPTool',
    'MCPResource',
    'MCPError',
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


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "mcp",
        "version": __version__,
        "description": "Model Context Protocol tool registration and discovery",
        "features": FEATURES,
    }
