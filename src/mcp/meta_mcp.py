"""
MCP (Model Context Protocol) Meta Tools

This module exposes MCP server's own metadata and status as tools for self-reflection
and server introspection. These meta-tools allow clients to query the server about
its own state, capabilities, and operational parameters.

Key Features:
- Server status and uptime information
- Authentication and encryption status
- Module loading statistics and diagnostics
- Tool and resource discovery helpers
- Performance metrics and error tracking
- Server configuration and capabilities introspection
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global server start time for uptime tracking
SERVER_START_TIME = time.time()

def get_mcp_server_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get comprehensive status information about the MCP server.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance running the server.
                          
    Returns:
        Dictionary containing detailed server status information.
    """
    uptime_seconds = time.time() - SERVER_START_TIME
    uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))
    
    # Get server status from the instance
    server_status = mcp_instance_ref.get_server_status()
    
    # Get module information
    modules_info = {}
    for name, module_info in mcp_instance_ref.modules.items():
        modules_info[name] = {
            "name": module_info.name,
            "path": str(module_info.path),
            "status": module_info.status,
            "tools_count": module_info.tools_count,
            "resources_count": module_info.resources_count,
            "load_time": module_info.load_time,
            "error_message": module_info.error_message
        }
    
    return {
        "success": True,
        "server_name": mcp_instance_ref.get_capabilities().get("name", "Unknown GNN MCP Server"),
        "server_version": mcp_instance_ref.get_capabilities().get("version", "Unknown"),
        "status": "running",
        "start_time_unix": SERVER_START_TIME,
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": uptime_str,
        "loaded_modules_count": len([m for m in mcp_instance_ref.modules.values() if m.status == "loaded"]),
        "failed_modules_count": len([m for m in mcp_instance_ref.modules.values() if m.status == "error"]),
        "total_modules": len(mcp_instance_ref.modules),
        "registered_tools_count": len(mcp_instance_ref.tools),
        "registered_resources_count": len(mcp_instance_ref.resources),
        "request_count": server_status.get("request_count", 0),
        "error_count": server_status.get("error_count", 0),
        "error_rate": server_status.get("error_rate", 0),
        "modules": modules_info,
        "performance": {
            "avg_execution_times": server_status.get("avg_execution_times", {}),
            "last_activity": server_status.get("last_activity", 0)
        }
    }

def get_mcp_auth_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get the authentication status and configuration of the MCP server.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance.
        
    Returns:
        Dictionary with authentication status and configuration.
    """
    return {
        "success": True,
        "authentication_type": "none_implemented",
        "access_level": "unrestricted_local_access",
        "transport_security": {
            "stdio": "local_process_only",
            "http": "local_network_only",
            "https": "not_configured"
        },
        "description": "Server does not implement explicit authentication. Relies on transport security.",
        "recommendations": [
            "Use stdio transport for local-only access",
            "Configure HTTPS for HTTP transport if needed",
            "Implement authentication if exposing to untrusted networks"
        ]
    }

def get_mcp_encryption_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get the encryption status of the MCP server connections and data handling.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance.

    Returns:
        Dictionary with encryption status and recommendations.
    """
    return {
        "success": True,
        "transport_encryption": {
            "stdio": {
                "encrypted": False,
                "description": "Plaintext communication within local process",
                "security_level": "high (local only)"
            },
            "http": {
                "encrypted": False,
                "description": "Plaintext HTTP communication",
                "security_level": "low (network accessible)"
            }
        },
        "data_at_rest": {
            "encrypted": False,
            "description": "Server is stateless, no persistent data storage"
        },
        "recommendations": [
            "Use stdio transport for maximum security",
            "Configure HTTPS for HTTP transport if needed",
            "Implement TLS certificates for production HTTP deployment"
        ]
    }

def get_mcp_module_info(mcp_instance_ref, module_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific loaded module.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance.
        module_name: Name of the module to query.
        
    Returns:
        Dictionary with detailed module information.
    """
    if module_name not in mcp_instance_ref.modules:
        return {
            "success": False,
            "error": f"Module '{module_name}' not found",
            "available_modules": list(mcp_instance_ref.modules.keys())
        }
    
    module_info = mcp_instance_ref.modules[module_name]
    
    # Get tools and resources from this module
    module_tools = []
    module_resources = []
    
    for tool_name, tool in mcp_instance_ref.tools.items():
        if tool.module == module_info.name:
            module_tools.append({
                "name": tool_name,
                "description": tool.description,
                "category": tool.category,
                "version": tool.version
            })
    
    for uri, resource in mcp_instance_ref.resources.items():
        if resource.module == module_info.name:
            module_resources.append({
                "uri": uri,
                "description": resource.description,
                "category": resource.category,
                "version": resource.version
            })
    
    return {
        "success": True,
        "module_name": module_name,
        "full_name": module_info.name,
        "path": str(module_info.path),
        "status": module_info.status,
        "load_time": module_info.load_time,
        "error_message": module_info.error_message,
        "tools": module_tools,
        "resources": module_resources,
        "tools_count": len(module_tools),
        "resources_count": len(module_resources)
    }

def get_mcp_tool_categories(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get tools organized by category for easier discovery.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance.
        
    Returns:
        Dictionary with tools organized by category.
    """
    categories = {}
    
    for tool_name, tool in mcp_instance_ref.tools.items():
        category = tool.category or "General"
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            "name": tool_name,
            "description": tool.description,
            "module": tool.module,
            "version": tool.version
        })
    
    return {
        "success": True,
        "categories": categories,
        "total_tools": len(mcp_instance_ref.tools),
        "total_categories": len(categories)
    }

def get_mcp_performance_metrics(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get performance metrics and statistics for the MCP server.
    
    Args:
        mcp_instance_ref: Reference to the MCP instance.
        
    Returns:
        Dictionary with performance metrics.
    """
    server_status = mcp_instance_ref.get_server_status()
    
    return {
        "success": True,
        "uptime": {
            "seconds": server_status.get("uptime_seconds", 0),
            "formatted": server_status.get("uptime_formatted", "Unknown")
        },
        "requests": {
            "total": server_status.get("request_count", 0),
            "errors": server_status.get("error_count", 0),
            "error_rate": server_status.get("error_rate", 0)
        },
        "execution_times": server_status.get("avg_execution_times", {}),
        "last_activity": server_status.get("last_activity", 0),
        "modules": {
            "loaded": server_status.get("modules_loaded", 0),
            "failed": server_status.get("modules_failed", 0),
            "total": server_status.get("modules_count", 0)
        }
    }

def register_tools(mcp_instance):
    """
    Register MCP meta-tools with the MCP server itself.
    
    These tools provide introspection and diagnostic capabilities for the MCP server,
    allowing clients to understand the server's state, capabilities, and performance.
    
    Args:
        mcp_instance: The main MCP instance to register tools with.
    """
    logger.info("Registering MCP meta-tools")
    
    # Create wrapper functions to pass the mcp_instance
    def get_status_wrapper():
        return get_mcp_server_status(mcp_instance)
        
    def get_auth_wrapper():
        return get_mcp_auth_status(mcp_instance)
        
    def get_encryption_wrapper():
        return get_mcp_encryption_status(mcp_instance)
    
    def get_module_info_wrapper(module_name: str):
        return get_mcp_module_info(mcp_instance, module_name)
    
    def get_tool_categories_wrapper():
        return get_mcp_tool_categories(mcp_instance)
    
    def get_performance_metrics_wrapper():
        return get_mcp_performance_metrics(mcp_instance)
    
    # Register meta-tools
    mcp_instance.register_tool(
        name="get_mcp_server_capabilities",
        func=mcp_instance.get_capabilities,
        schema={},
        description="Retrieves the full capabilities description of this MCP server, including all tools and resources.",
        module="meta",
        category="Server Info",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_server_status",
        func=get_status_wrapper,
        schema={},
        description="Provides comprehensive operational status of the MCP server, including uptime, modules, and performance metrics.",
        module="meta",
        category="Server Info",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_server_auth_status",
        func=get_auth_wrapper,
        schema={},
        description="Describes the current authentication mechanisms and security configuration of the MCP server.",
        module="meta",
        category="Security",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_server_encryption_status",
        func=get_encryption_wrapper,
        schema={},
        description="Describes the current encryption status for server transport and data handling with security recommendations.",
        module="meta",
        category="Security",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_module_info",
        func=get_module_info_wrapper,
        schema={
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "description": "Name of the module to query"
                }
            },
            "required": ["module_name"]
        },
        description="Get detailed information about a specific loaded module, including its tools and resources.",
        module="meta",
        category="Diagnostics",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_tool_categories",
        func=get_tool_categories_wrapper,
        schema={},
        description="Get tools organized by category for easier discovery and navigation.",
        module="meta",
        category="Discovery",
        version="1.0.0"
    )
    
    mcp_instance.register_tool(
        name="get_mcp_performance_metrics",
        func=get_performance_metrics_wrapper,
        schema={},
        description="Get performance metrics and statistics for the MCP server, including execution times and error rates.",
        module="meta",
        category="Performance",
        version="1.0.0"
    )
    
    logger.info("Successfully registered MCP meta-tools") 