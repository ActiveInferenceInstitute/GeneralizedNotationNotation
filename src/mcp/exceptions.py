#!/usr/bin/env python3
"""
MCP Exceptions - Error Classes for Model Context Protocol

This module defines all MCP-specific exception classes with enhanced context tracking.
These exceptions provide structured error information for MCP operations.
"""

import time
import traceback
from typing import Dict, List, Any, Optional


class MCPError(Exception):
    """Enhanced base class for MCP related errors with better context tracking."""
    def __init__(self, message: str, code: int = -32000, data: Optional[Any] = None, 
                 tool_name: Optional[str] = None, module_name: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.data = data or {}
        self.tool_name = tool_name
        self.module_name = module_name
        self.timestamp = time.time()
        
        # Add context information
        if tool_name:
            self.data["tool_name"] = tool_name
        if module_name:
            self.data["module_name"] = module_name


class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""
    def __init__(self, tool_name: str, available_tools: Optional[List[str]] = None):
        super().__init__(
            f"Tool '{tool_name}' not found", 
            code=-32601,
            data={"available_tools": available_tools or []},
            tool_name=tool_name
        )


class MCPResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""
    def __init__(self, uri: str, available_resources: Optional[List[str]] = None):
        super().__init__(
            f"Resource '{uri}' not found", 
            code=-32601,
            data={"available_resources": available_resources or []},
            tool_name=uri
        )


class MCPInvalidParamsError(MCPError):
    """Raised when tool parameters are invalid."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 tool_name: Optional[str] = None, schema: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, 
            code=-32602, 
            data={"details": details or {}, "schema": schema},
            tool_name=tool_name
        )


class MCPToolExecutionError(MCPError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, original_exception: Exception, 
                 execution_time: Optional[float] = None):
        super().__init__(
            f"Tool '{tool_name}' execution failed: {str(original_exception)}",
            code=-32603,
            data={
                "original_exception": str(original_exception), 
                "traceback": traceback.format_exc(),
                "execution_time": execution_time
            },
            tool_name=tool_name
        )


class MCPSDKNotFoundError(MCPError):
    """Raised when required SDK is not found."""
    def __init__(self, message: str = "MCP SDK not found or failed to initialize.", 
                 sdk_paths: Optional[List[str]] = None):
        super().__init__(
            message, 
            code=-32001,
            data={"sdk_paths": sdk_paths or []}
        )


class MCPValidationError(MCPError):
    """Raised when validation fails."""
    def __init__(self, message: str, field: Optional[str] = None, 
                 tool_name: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(
            message, 
            code=-32602, 
            data={"field": field, "value": value},
            tool_name=tool_name
        )


class MCPModuleLoadError(MCPError):
    """Raised when a module fails to load."""
    def __init__(self, module_name: str, original_exception: Exception):
        super().__init__(
            f"Module '{module_name}' failed to load: {str(original_exception)}",
            code=-32003,
            data={
                "original_exception": str(original_exception),
                "traceback": traceback.format_exc()
            },
            module_name=module_name
        )


class MCPPerformanceError(MCPError):
    """Raised when performance thresholds are exceeded."""
    def __init__(self, operation: str, execution_time: float, threshold: float):
        super().__init__(
            f"Performance threshold exceeded for '{operation}': {execution_time:.3f}s > {threshold:.3f}s",
            code=-32004,
            data={
                "operation": operation,
                "execution_time": execution_time,
                "threshold": threshold
            }
        )


class MCPRateLimitError(MCPError):
    """Raised when rate limit is exceeded for a tool."""
    def __init__(self, tool_name: str, rate_limit: float, current_rate: float):
        super().__init__(
            f"Rate limit exceeded for tool '{tool_name}': {current_rate:.2f} req/s > {rate_limit:.2f} req/s",
            code=-32005,
            data={
                "tool_name": tool_name,
                "rate_limit": rate_limit,
                "current_rate": current_rate
            },
            tool_name=tool_name
        )


class MCPCacheError(MCPError):
    """Raised when cache operations fail."""
    def __init__(self, operation: str, cache_key: str, original_error: Exception):
        super().__init__(
            f"Cache operation '{operation}' failed for key '{cache_key}': {str(original_error)}",
            code=-32006,
            data={
                "operation": operation,
                "cache_key": cache_key,
                "original_error": str(original_error)
            }
        )


class MCPModuleDiscoveryError(MCPError):
    """Raised when module discovery fails."""
    def __init__(self, module_name: str, discovery_path: str, original_error: Exception):
        super().__init__(
            f"Module discovery failed for '{module_name}' in '{discovery_path}': {str(original_error)}",
            code=-32007,
            data={
                "module_name": module_name,
                "discovery_path": discovery_path,
                "original_error": str(original_error)
            },
            module_name=module_name
        )


# Export all exceptions
__all__ = [
    "MCPError",
    "MCPToolNotFoundError",
    "MCPResourceNotFoundError",
    "MCPInvalidParamsError",
    "MCPToolExecutionError",
    "MCPSDKNotFoundError",
    "MCPValidationError",
    "MCPModuleLoadError",
    "MCPPerformanceError",
    "MCPRateLimitError",
    "MCPCacheError",
    "MCPModuleDiscoveryError",
]
