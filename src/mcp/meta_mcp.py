"""
MCP (Model Context Protocol) integration for self-reflection and server metadata.

This module exposes MCP server's own metadata and status as tools.
It is intended to be loaded by the main MCP instance itself.
"""

import os
import time
from typing import Dict, Any

# This module will be imported by the main mcp_instance, so it can access it.
# We need a way to get a reference to the main mcp_instance that is loading this module.
# This is a bit circular, but the `register_tools` function will be passed the mcp_instance.

SERVER_START_TIME = time.time()

# MCP Tools for MCP Meta Module

def get_mcp_server_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get the current status of the MCP server itself.
    
    Args:
        mcp_instance_ref: A reference to the MCP instance running the server.
                          This is passed by the registration mechanism.
                          
    Returns:
        Dictionary server status information.
    """
    uptime_seconds = time.time() - SERVER_START_TIME
    uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))
    
    loaded_modules = list(mcp_instance_ref.modules.keys())
    tool_count = len(mcp_instance_ref.tools)
    resource_count = len(mcp_instance_ref.resources)
    
    return {
        "success": True,
        "server_name": mcp_instance_ref.get_capabilities().get("name", "Unknown GNN MCP Server"),
        "server_version": mcp_instance_ref.get_capabilities().get("version", "Unknown"),
        "status": "running",
        "start_time_unix": SERVER_START_TIME,
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": uptime_str,
        "loaded_modules_count": len(loaded_modules),
        "loaded_modules": loaded_modules,
        "registered_tools_count": tool_count,
        "registered_resources_count": resource_count,
        # In a real scenario, you might add request counts, error rates etc.
    }

def get_mcp_auth_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get the authentication status of the MCP server.
    
    Args:
        mcp_instance_ref: A reference to the MCP instance.
        
    Returns:
        Dictionary with authentication status.
    """
    # Currently, no specific auth is implemented beyond transport (stdio, http local)
    return {
        "success": True,
        "authentication_type": "none_implemented",
        "access_level": "unrestricted_local_access",
        "description": "Server does not implement explicit authentication. Relies on transport security."
    }

def get_mcp_encryption_status(mcp_instance_ref) -> Dict[str, Any]:
    """
    Get the encryption status of the MCP server connections.
    
    Args:
        mcp_instance_ref: A reference to the MCP instance.

    Returns:
        Dictionary with encryption status.
    """
    # Currently, stdio is unencrypted by default. HTTP is unencrypted by default.
    # HTTPS would need to be explicitly configured.
    return {
        "success": True,
        "stdio_transport_encryption": "none (plaintext)",
        "http_transport_encryption": "none (plaintext, HTTPS not configured by default)",
        "data_at_rest_encryption": "not_applicable (server is stateless or relies on filesystem)"
    }

# MCP Registration Function
def register_tools(mcp_instance): # This mcp_instance IS the main server's MCP instance
    """Register MCP meta-tools with the MCP server itself."""
    
    # To call these tools, the mcp_instance needs to be passed. 
    # The MCP standard doesn't have a built-in way for a tool to get a reference to its host server.
    # So, we'll wrap the tool functions to pass the mcp_instance.
    
    def get_status_wrapper():
        return get_mcp_server_status(mcp_instance)
        
    def get_auth_wrapper():
        return get_mcp_auth_status(mcp_instance)
        
    def get_encryption_wrapper():
        return get_mcp_encryption_status(mcp_instance)
        
    mcp_instance.register_tool(
        "get_mcp_server_capabilities", # Renaming from get_capabilities to avoid direct name clash if called by tool name
        mcp_instance.get_capabilities, # Directly register the existing method
        {},
        "Retrieves the full capabilities description of this MCP server, including all tools and resources."
    )
    
    mcp_instance.register_tool(
        "get_mcp_server_status",
        get_status_wrapper, 
        {},
        "Provides the current operational status of the MCP server, including uptime and loaded modules."
    )
    
    mcp_instance.register_tool(
        "get_mcp_server_auth_status",
        get_auth_wrapper,
        {},
        "Describes the current authentication mechanisms and status of the MCP server."
    )
    
    mcp_instance.register_tool(
        "get_mcp_server_encryption_status",
        get_encryption_wrapper,
        {},
        "Describes the current encryption status for server transport and data handling."
    ) 