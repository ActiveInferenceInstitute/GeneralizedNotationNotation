#!/usr/bin/env python3
"""Server introspection and lookup for the GNN MCP server.

Mechanical extraction from ``gnn.mcp.mcp`` (MAJ-04 sibling-mixin split):
``MCPIntrospectionMixin`` holds the verbatim capabilities/status/info
methods plus the validation and URI-template helpers, and ``MCP`` in
``mcp.py`` inherits from it, so every response shape and dict key is
unchanged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ._late_binding import _MCPModuleRef
from .exceptions import MCPResourceNotFoundError
from .models import MCPModuleInfo, MCPPerformanceMetrics, MCPResource, MCPTool
from .validation import _validate_params

# Configure logging
logger = logging.getLogger("mcp")

# Late-bound clock: attribute access resolves through ``gnn.mcp.mcp`` so the
# fake-clock swap in tests/mcp/test_registry_internals.py keeps working.
time = _MCPModuleRef("time")


class MCPIntrospectionMixin:
    """Verbatim introspection methods moved from ``MCP``."""

    if TYPE_CHECKING:
        # Shared ``MCP`` state the moved bodies touch (see discovery.py).
        _lock: Any
        _request_count: int
        _error_count: int
        _last_activity: float
        _strict_validation: bool
        tools: Dict[str, MCPTool]
        resources: Dict[str, MCPResource]
        modules: Dict[str, MCPModuleInfo]
        _tool_execution_times: Dict[str, List[float]]
        _performance_metrics: MCPPerformanceMetrics

        # Facade assembly property (uptime lives on ``MCP``).
        @property
        def uptime(self) -> float: ...

    def get_resource(self, uri: str) -> Dict[str, Any]:
        """
        Retrieve a resource by URI.

        Args:
            uri: URI of the resource to retrieve

        Returns:
            Resource content

        Raises:
            MCPResourceNotFoundError: If resource is not found
        """
        with self._lock:
            self._request_count += 1
            self._last_activity = time.time()

            # Find matching resource
            matching_resource = None
            for resource_template, resource in self.resources.items():
                if self._match_uri_template(resource_template, uri):
                    matching_resource = resource
                    break

            if not matching_resource:
                raise MCPResourceNotFoundError(uri)

            # Enforce auth gate — mirrors execute_tool() pattern
            if matching_resource.requires_auth:
                raise MCPResourceNotFoundError(uri)

            try:
                # Retrieve resource content
                content = matching_resource.retriever(uri)

                # Add metadata
                result: dict[str, Any] = {
                    "content": content,
                    "uri": uri,
                    "mime_type": matching_resource.mime_type,
                    "cacheable": matching_resource.cacheable,
                    "retrieved_at": time.time(),
                }

                logger.debug(f"Resource '{uri}' retrieved successfully")
                return result

            except Exception as e:
                logger.error(f"Resource '{uri}' retrieval failed: {e}")
                raise MCPResourceNotFoundError(uri) from e

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities including all available tools and resources."""
        with self._lock:
            tools_list: list[Any] = []
            for tool in self.tools.values():
                tools_list.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.schema,
                        "module": tool.module,
                        "category": tool.category,
                        "version": tool.version,
                        "tags": tool.tags,
                        "examples": tool.examples,
                        "experimental": tool.experimental,
                        "timeout": tool.timeout,
                        "max_concurrent": tool.max_concurrent,
                        "requires_auth": tool.requires_auth,
                        "rate_limit": tool.rate_limit,
                        "cache_ttl": tool.cache_ttl,
                        "input_validation": tool.input_validation,
                        "output_validation": tool.output_validation,
                    }
                )

            resources_list: list[Any] = []
            for resource in self.resources.values():
                resources_list.append(
                    {
                        "uri_template": resource.uri_template,
                        "description": resource.description,
                        "module": resource.module,
                        "category": resource.category,
                        "version": resource.version,
                        "mime_type": resource.mime_type,
                        "cacheable": resource.cacheable,
                        "tags": resource.tags,
                        "timeout": resource.timeout,
                        "requires_auth": resource.requires_auth,
                        "rate_limit": resource.rate_limit,
                        "cache_ttl": resource.cache_ttl,
                        "compression": resource.compression,
                        "encryption": resource.encryption,
                    }
                )

            return {
                "tools": tools_list,
                "resources": resources_list,
                "validation_mode": (
                    "strict" if self._strict_validation else "required_only"
                ),
                "server": {
                    "name": "GNN MCP Server",
                    "version": "1.0.0",
                    "description": "Model Context Protocol server for GeneralizedNotationNotation",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": True},
                    },
                },
            }

    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status information."""
        with self._lock:
            uptime_seconds = self.uptime
            uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))

            # Calculate tool categories
            categories: Any = defaultdict(int)
            for tool in self.tools.values():
                categories[tool.category or "uncategorized"] += 1

            # Calculate resource categories
            resource_categories: Any = defaultdict(int)
            for resource in self.resources.values():
                resource_categories[resource.category or "uncategorized"] += 1

            return {
                "uptime": uptime_seconds,
                "uptime_formatted": uptime_str,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "tools_count": len(self.tools),
                "resources_count": len(self.resources),
                "modules_count": len(self.modules),
                "last_activity": self._last_activity,
                "tool_categories": dict(categories),
                "resource_categories": dict(resource_categories),
                "performance_metrics": {
                    "total_requests": self._performance_metrics.total_requests,
                    "successful_requests": self._performance_metrics.successful_requests,
                    "failed_requests": self._performance_metrics.failed_requests,
                    "average_execution_time": self._performance_metrics.average_execution_time,
                    "max_execution_time": self._performance_metrics.max_execution_time,
                    "min_execution_time": self._performance_metrics.min_execution_time
                    if self._performance_metrics.min_execution_time != float("inf")
                    else 0.0,
                },
            }

    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific module."""
        with self._lock:
            if module_name not in self.modules:
                return None

            module_info = self.modules[module_name]

            # Get tools for this module
            module_tools = [
                tool.name for tool in self.tools.values() if tool.module == module_name
            ]

            # Get resources for this module
            module_resources = [
                resource.uri_template
                for resource in self.resources.values()
                if resource.module == module_name
            ]

            return {
                "name": module_info.name,
                "path": str(module_info.path),
                "status": module_info.status,
                "tools_count": module_info.tools_count,
                "resources_count": module_info.resources_count,
                "load_time": module_info.load_time,
                "version": module_info.version,
                "description": module_info.description,
                "dependencies": module_info.dependencies,
                "last_updated": module_info.last_updated,
                "error_message": module_info.error_message,
                "tools": module_tools,
                "resources": module_resources,
            }

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        with self._lock:
            if tool_name not in self.tools:
                return None

            tool = self.tools[tool_name]

            # Get execution statistics
            execution_times = self._tool_execution_times.get(tool_name, [])
            avg_execution_time = (
                sum(execution_times) / len(execution_times) if execution_times else 0.0
            )

            return {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
                "module": tool.module,
                "category": tool.category,
                "version": tool.version,
                "tags": tool.tags,
                "examples": tool.examples,
                "experimental": tool.experimental,
                "usage_count": self._performance_metrics.tool_usage_stats.get(
                    tool_name, 0
                ),
                "average_execution_time": avg_execution_time,
                "execution_count": len(execution_times),
                "timeout": tool.timeout,
                "max_concurrent": tool.max_concurrent,
                "requires_auth": tool.requires_auth,
                "rate_limit": tool.rate_limit,
                "cache_ttl": tool.cache_ttl,
                "input_validation": tool.input_validation,
                "output_validation": tool.output_validation,
            }

    def _validate_params(self, schema: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Validate tool parameters against their JSON schema.

        Thin delegating wrapper over the stateless validators in
        ``gnn.mcp.validation`` (STR-2 phase 1 extraction); signature and
        error behavior are unchanged for all call sites.

        Args:
            schema: JSON schema for validation
            params: Parameters to validate

        Raises:
            MCPValidationError: If validation fails
        """
        _validate_params(schema, params, strict=self._strict_validation)

    def _match_uri_template(self, template: str, uri: str) -> bool:
        """Check if URI matches template pattern."""
        # Simple template matching - can be enhanced with regex
        if template == uri:
            return True

        # Handle simple {param} patterns
        if "{" in template and "}" in template:
            # This is a simplified implementation
            # In a real implementation, you'd want more sophisticated pattern matching
            template_parts = template.split("/")
            uri_parts = uri.split("/")

            if len(template_parts) != len(uri_parts):
                return False

            for template_part, uri_part in zip(template_parts, uri_parts):
                if template_part.startswith("{") and template_part.endswith("}"):
                    continue
                if template_part != uri_part:
                    return False

            return True

        return False

    def _validate_output(self, result: Any) -> Any:
        """Validate tool output against its declared contract.

        ``MCPTool`` declares no output schema today, so there is no contract
        to validate against: any return value — including ``None`` — passes.
        (The previous behavior rejected ``None`` with -32602 INVALID_PARAMS
        even though the params were valid and no contract existed; tools
        returning ``None`` are legitimate.) Keep the method as the extension
        point for a future ``returns`` schema.
        """
        return result
