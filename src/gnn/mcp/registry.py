#!/usr/bin/env python3
"""Tool and resource registry for the GNN MCP server.

Mechanical extraction from ``gnn.mcp.mcp`` (MAJ-04 sibling-mixin split):
``MCPRegistryMixin`` holds the verbatim registration/listing methods and
``MCP`` in ``mcp.py`` inherits from it, so every import path, method
resolution, and registered-tool name is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from .exceptions import MCPInvalidParamsError
from .models import MCPResource, MCPTool

# Configure logging
logger = logging.getLogger("mcp")


class MCPRegistryMixin:
    """Verbatim registry methods moved from ``MCP``."""

    if TYPE_CHECKING:
        # Shared ``MCP`` state the moved bodies touch (see discovery.py).
        _lock: Any
        tools: Dict[str, MCPTool]
        resources: Dict[str, MCPResource]
        _tools_list_cache: Optional[List[Dict[str, Any]]]
        _tools_names_cache: Optional[List[str]]

        # Members owned by the facade assembly / discovery mixin that the
        # moved bodies call through ``self``.
        @classmethod
        def _normalize_tool_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]: ...
        def _default_tool_metadata(
            self, module: str, category: str
        ) -> Tuple[str, str]: ...

    # --- Compatibility: simple listings used by reporting and diagnostics ---
    def list_available_tools(
        self, include_metadata: bool = True
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Return a list of registered tools. If include_metadata is True, returns a list of
        dictionaries with metadata; otherwise returns tool names.
        """
        with self._lock:
            if include_metadata:
                if self._tools_list_cache is None:
                    result: List[Dict[str, Any]] = []
                    for name, tool in self.tools.items():
                        result.append(
                            {
                                "name": name,
                                "description": getattr(tool, "description", ""),
                                "module": getattr(tool, "module", ""),
                                "category": getattr(tool, "category", ""),
                                "version": getattr(tool, "version", "1.0.0"),
                            }
                        )
                    self._tools_list_cache = sorted(result, key=lambda t: t["name"])
                return self._tools_list_cache
            else:
                if self._tools_names_cache is None:
                    self._tools_names_cache = sorted(self.tools.keys())
                return self._tools_names_cache

    def list_available_resources(
        self, include_metadata: bool = True
    ) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Return a list of registered resources. If include_metadata is True, returns a list of
        dictionaries with metadata; otherwise returns resource URIs.
        """
        with self._lock:
            if include_metadata:
                result: List[Dict[str, Any]] = []
                for uri, res in self.resources.items():
                    result.append(
                        {
                            "uri": uri,
                            "description": getattr(res, "description", ""),
                            "module": getattr(res, "module", ""),
                            "category": getattr(res, "category", ""),
                            "version": getattr(res, "version", "1.0.0"),
                        }
                    )
                return sorted(result, key=lambda r: r["uri"])
            else:
                return sorted(self.resources.keys())

    def register_tool(
        self,
        name: str,
        func: Optional[Callable] = None,
        schema: Optional[Dict[str, Any]] = None,
        description: str = "",
        module: str = "",
        category: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        experimental: bool = False,
        timeout: Optional[float] = None,
        max_concurrent: int = 1,
        requires_auth: bool = False,
        rate_limit: Optional[float] = None,
        cache_ttl: Optional[float] = None,
        input_validation: bool = True,
        output_validation: bool = True,
        # Alternate metadata keywords accepted by module mcp files.
        function: Optional[Callable] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        returns: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a new tool with the MCP server.

        Args:
            name: Unique name for the tool
            func: Callable function to execute
            schema: JSON schema for tool parameters
            description: Human-readable description
            module: Module name that provides this tool
            category: Tool category for organization
            version: Tool version
            tags: List of tags for categorization
            examples: List of example parameter sets
            experimental: Whether the tool is experimental
            timeout: Optional timeout for the tool in seconds
            max_concurrent: Maximum number of concurrent executions
            requires_auth: Whether the tool requires authentication
            rate_limit: Optional rate limit for the tool in requests per second
            cache_ttl: Optional cache TTL for the tool in seconds
            input_validation: Whether to validate input parameters
            output_validation: Whether to validate output results
        """
        with self._lock:
            # Validate inputs before proceeding
            if not name or not isinstance(name, str):
                raise MCPInvalidParamsError("Tool name must be a non-empty string")

            if func is None and function is not None:
                func = function
            elif func is None:
                raise MCPInvalidParamsError("Tool function is required")

            if not callable(func):
                raise MCPInvalidParamsError("Tool function must be callable")

            if schema is None:
                schema = {}

            # Validate schema
            if not isinstance(schema, dict):
                raise MCPInvalidParamsError("Tool schema must be a dictionary")

            if name in self.tools:
                # Explicit duplicate: warn and overwrite. Overwriting is
                # back-compat (re-registration on module reload relies on
                # last-write-wins, e.g. test_register_tool_overwrites), but
                # the replacement is now visible instead of a silent
                # debug-log no-op.
                logger.warning(
                    f"Tool '{name}' already registered; overwriting previous registration"
                )

            # Convert older "parameters" list format into JSON schema if provided
            if parameters and not schema:
                props: Dict[str, Any] = {}
                required_fields: List[str] = []
                type_map: dict[str, Any] = {
                    "string": "string",
                    "boolean": "boolean",
                    "integer": "integer",
                    "number": "number",
                    "array": "array",
                    "object": "object",
                }
                for p in parameters:
                    pname = str(p.get("name") or p.get("param"))
                    ptype = type_map.get(p.get("type", "string"), "string")
                    prop: Dict[str, Any] = {"type": ptype}
                    if "description" in p:
                        prop["description"] = p["description"]
                    if "enum" in p:
                        prop["enum"] = p["enum"]
                    if "default" in p:
                        prop["default"] = p["default"]
                    props[pname] = prop
                    if p.get("required", False):
                        required_fields.append(pname)
                schema = {"type": "object", "properties": props}
                if required_fields:
                    schema["required"] = required_fields

            schema = self._normalize_tool_schema(schema)
            module, category = self._default_tool_metadata(module, category)

            tool = MCPTool(
                name=name,
                func=func,
                schema=schema,
                description=description,
                module=module,
                category=category,
                version=version,
                tags=tags or [],
                examples=examples or [],
                experimental=experimental,
                timeout=timeout,
                max_concurrent=max_concurrent,
                requires_auth=requires_auth,
                rate_limit=rate_limit,
                cache_ttl=cache_ttl,
                input_validation=input_validation,
                output_validation=output_validation,
            )

            self.tools[name] = tool
            self._tools_list_cache = None
            self._tools_names_cache = None
            logger.debug(f"Registered tool: {name}")

    def register_resource(
        self,
        uri_template: str,
        retriever: Callable,
        description: str,
        module: str = "",
        category: str = "",
        version: str = "1.0.0",
        mime_type: str = "application/json",
        cacheable: bool = True,
        tags: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        requires_auth: bool = False,
        rate_limit: Optional[float] = None,
        cache_ttl: Optional[float] = None,
        compression: bool = False,
        encryption: bool = False,
    ) -> Any:
        """
        Register a new resource with the MCP server.

        Args:
            uri_template: URI template for the resource
            retriever: Function to retrieve resource content
            description: Human-readable description
            module: Module name that provides this resource
            category: Resource category for organization
            version: Resource version
            mime_type: MIME type of the resource
            cacheable: Whether the resource can be cached
            tags: List of tags for categorization
            timeout: Optional timeout for the resource in seconds
            requires_auth: Whether the resource requires authentication
            rate_limit: Optional rate limit for the resource in requests per second
            cache_ttl: Optional cache TTL for the resource in seconds
            compression: Whether the resource is compressed
            encryption: Whether the resource is encrypted
        """
        with self._lock:
            module, category = self._default_tool_metadata(module, category)

            if uri_template in self.resources:
                logger.warning(
                    f"Resource '{uri_template}' already registered, overwriting"
                )

            resource = MCPResource(
                uri_template=uri_template,
                retriever=retriever,
                description=description,
                module=module,
                category=category,
                version=version,
                mime_type=mime_type,
                cacheable=cacheable,
                tags=tags or [],
                timeout=timeout,
                requires_auth=requires_auth,
                rate_limit=rate_limit,
                cache_ttl=cache_ttl,
                compression=compression,
                encryption=encryption,
            )

            self.resources[uri_template] = resource
            logger.debug(f"Registered resource: {uri_template}")
