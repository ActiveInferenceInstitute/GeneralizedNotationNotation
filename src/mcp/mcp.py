#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Core Implementation for GNN

This module provides the core MCP server implementation for the GeneralizedNotationNotation (GNN) project.
It handles tool discovery, registration, and execution across all GNN modules.

The MCP server exposes GNN functionalities as standardized tools that can be accessed by
MCP-compatible clients such as AI assistants, IDEs, and automated research pipelines.

Key Features:
- Dynamic module discovery and tool registration
- JSON-RPC 2.0 compliant request/response handling
- Comprehensive error handling and logging
- Support for both stdio and HTTP transport layers
- Extensible architecture for adding new tools and resources
"""

import importlib
import os
import sys
from pathlib import Path
import logging
import inspect
import json
import time
from typing import Dict, List, Any, Callable, Optional, TypedDict, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger("mcp")

# --- Custom MCP Exceptions ---
class MCPError(Exception):
    """Base class for MCP related errors."""
    def __init__(self, message: str, code: int = -32000, data: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.data = data

class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found.", code=-32601, data=f"Tool '{tool_name}' not found.")

class MCPResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""
    def __init__(self, uri: str):
        super().__init__(f"Resource '{uri}' not found.", code=-32601, data=f"Resource '{uri}' not found.")

class MCPInvalidParamsError(MCPError):
    """Raised when tool parameters are invalid."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=-32602, data=details)

class MCPToolExecutionError(MCPError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, original_exception: Exception):
        super().__init__(
            f"Error executing tool '{tool_name}': {original_exception}", 
            code=-32000, 
            data=str(original_exception)
        )

class MCPSDKNotFoundError(MCPError):
    """Raised when MCP SDK is not found or fails to initialize."""
    def __init__(self, message: str = "MCP SDK not found or failed to initialize."):
        super().__init__(message, code=-32001, data=message)

class MCPValidationError(MCPError):
    """Raised when request validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, code=-32602, data={"field": field} if field else None)

# --- MCP Data Structures ---
@dataclass
class MCPTool:
    """Represents an MCP tool that can be executed."""
    name: str
    func: Callable
    schema: Dict[str, Any]
    description: str
    module: str = ""
    category: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate tool configuration after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.func):
            raise ValueError("Tool function must be callable")
        if not isinstance(self.schema, dict):
            raise ValueError("Tool schema must be a dictionary")

@dataclass
class MCPResource:
    """Represents an MCP resource that can be accessed."""
    uri_template: str
    retriever: Callable
    description: str
    module: str = ""
    category: str = ""
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate resource configuration after initialization."""
        if not self.uri_template:
            raise ValueError("Resource URI template cannot be empty")
        if not callable(self.retriever):
            raise ValueError("Resource retriever must be callable")

@dataclass
class MCPModuleInfo:
    """Information about a discovered MCP module."""
    name: str
    path: Path
    tools_count: int = 0
    resources_count: int = 0
    status: str = "loaded"
    error_message: Optional[str] = None
    load_time: float = 0.0

# --- MCP SDK Status Management ---
class MCPSDKStatus:
    """Manages MCP SDK status and configuration."""
    
    def __init__(self):
        self.found: bool = True
        self.details: str = "Using project's internal MCP implementation."
        self.version: str = "1.0.0"
        self.features: List[str] = ["tool_registration", "resource_access", "module_discovery"]
        self.last_check: float = time.time()
    
    def check_status(self) -> bool:
        """Check current SDK status."""
        # In a real implementation, this would check for actual SDK availability
        # For now, we assume the internal implementation is always available
        self.last_check = time.time()
        return self.found
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "found": self.found,
            "details": self.details,
            "version": self.version,
            "features": self.features,
            "last_check": self.last_check
        }

# Global SDK status instance
_MCP_SDK_STATUS = MCPSDKStatus()

# --- Main MCP Class ---
class MCP:
    """
    Main Model Context Protocol implementation.
    
    This class provides the core functionality for:
    - Discovering and loading MCP modules
    - Registering tools and resources
    - Executing tools and retrieving resources
    - Managing server capabilities and status
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.modules: Dict[str, MCPModuleInfo] = {}
        self._modules_discovered = False
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        
        # Performance tracking
        self._tool_execution_times: Dict[str, List[float]] = {}
        self._last_activity = time.time()
        
        logger.info("MCP server initialized")
    
    @property
    def uptime(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self._start_time
    
    @property
    def request_count(self) -> int:
        """Get total number of requests processed."""
        return self._request_count
    
    @property
    def error_count(self) -> int:
        """Get total number of errors encountered."""
        return self._error_count
    
    def discover_modules(self) -> bool:
        """
        Discover and load MCP modules from other directories.
        
        This method scans the src/ directory for modules with mcp.py files
        and loads them to register their tools and resources.
        
        Returns:
            bool: True if all modules loaded successfully, False otherwise.
        """
        if self._modules_discovered:
            logger.debug("MCP modules already discovered. Skipping redundant discovery.")
            return True

        root_dir = Path(__file__).parent.parent
        logger.info(f"Discovering MCP modules in {root_dir}")
        all_modules_loaded_successfully = True
        
        # Track discovery performance
        discovery_start = time.time()
        
        for directory in root_dir.iterdir():
            if not directory.is_dir() or directory.name.startswith('_'):
                continue
                
            mcp_file = directory / "mcp.py"
            if not mcp_file.exists():
                logger.debug(f"No MCP module found in {directory}")
                continue
                
            module_name = f"src.{directory.name}.mcp"
            module_start = time.time()
            
            try:
                # Add parent directory to path if needed
                if str(root_dir.parent) not in sys.path:
                    sys.path.append(str(root_dir.parent))
                
                module = importlib.import_module(module_name)
                logger.debug(f"Loaded MCP module: {module_name}")
                
                # Special handling for llm module initialization
                if module_name == "src.llm.mcp":
                    if hasattr(module, "initialize_llm_module") and callable(module.initialize_llm_module):
                        logger.debug(f"Calling initialize_llm_module for {module_name}")
                        module.initialize_llm_module(self)
                    else:
                        logger.warning(f"Module {module_name} does not have a callable initialize_llm_module function.")

                # Register tools and resources from the module
                if hasattr(module, "register_tools") and callable(module.register_tools):
                    tools_before = len(self.tools)
                    resources_before = len(self.resources)
                    
                    module.register_tools(self)
                    
                    tools_added = len(self.tools) - tools_before
                    resources_added = len(self.resources) - resources_before
                    
                    module_load_time = time.time() - module_start
                    
                    # Create module info
                    self.modules[directory.name] = MCPModuleInfo(
                        name=module_name,
                        path=mcp_file,
                        tools_count=tools_added,
                        resources_count=resources_added,
                        status="loaded",
                        load_time=module_load_time
                    )
                    
                    logger.info(f"Successfully loaded module {directory.name}: {tools_added} tools, {resources_added} resources")
                else:
                    logger.warning(f"Module {module_name} found but has no register_tools function.")
                    self.modules[directory.name] = MCPModuleInfo(
                        name=module_name,
                        path=mcp_file,
                        status="no_register_function"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load MCP module {module_name}: {str(e)}")
                all_modules_loaded_successfully = False
                
                self.modules[directory.name] = MCPModuleInfo(
                    name=module_name,
                    path=mcp_file,
                    status="error",
                    error_message=str(e)
                )

        # Special handling for core MCP tools in the mcp directory itself
        mcp_dir = Path(__file__).parent
        logger.debug(f"Discovering core MCP tools in {mcp_dir}")
        
        # Load SymPy MCP integration
        sympy_mcp_file = mcp_dir / "sympy_mcp.py"
        if sympy_mcp_file.exists():
            try:
                module = importlib.import_module("src.mcp.sympy_mcp")
                logger.debug(f"Loaded core MCP module: src.mcp.sympy_mcp")
                
                if hasattr(module, "register_tools") and callable(module.register_tools):
                    module.register_tools(self)
                    logger.info("Successfully registered SymPy MCP tools")
                
                self.modules["sympy_mcp"] = MCPModuleInfo(
                    name="src.mcp.sympy_mcp",
                    path=sympy_mcp_file,
                    status="loaded"
                )
            except Exception as e:
                logger.error(f"Failed to load core MCP module src.mcp.sympy_mcp: {str(e)}")
                all_modules_loaded_successfully = False
                
                self.modules["sympy_mcp"] = MCPModuleInfo(
                    name="src.mcp.sympy_mcp",
                    path=sympy_mcp_file,
                    status="error",
                    error_message=str(e)
                )

        discovery_time = time.time() - discovery_start
        logger.info(f"Module discovery completed in {discovery_time:.2f}s: {len(self.modules)} modules, {len(self.tools)} tools, {len(self.resources)} resources")
        
        self._modules_discovered = True
        return all_modules_loaded_successfully
    
    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any], description: str, 
                     module: str = "", category: str = "", version: str = "1.0.0"):
        """
        Register a new tool with the MCP.
        
        Args:
            name: Unique tool name
            func: Callable function to execute
            schema: JSON schema for tool parameters
            description: Human-readable description
            module: Source module name
            category: Tool category
            version: Tool version
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered. Overwriting.")
        
        try:
            tool = MCPTool(
                name=name,
                func=func,
                schema=schema,
                description=description,
                module=module,
                category=category,
                version=version
            )
            self.tools[name] = tool
            logger.debug(f"Registered tool: {name}")
        except Exception as e:
            logger.error(f"Failed to register tool {name}: {e}")
            raise
        
    def register_resource(self, uri_template: str, retriever: Callable, description: str,
                         module: str = "", category: str = "", version: str = "1.0.0"):
        """
        Register a new resource with the MCP.
        
        Args:
            uri_template: URI template for the resource
            retriever: Function to retrieve resource content
            description: Human-readable description
            module: Source module name
            category: Resource category
            version: Resource version
        """
        if uri_template in self.resources:
            logger.warning(f"Resource '{uri_template}' already registered. Overwriting.")
            
        try:
            resource = MCPResource(
                uri_template=uri_template,
                retriever=retriever,
                description=description,
                module=module,
                category=category,
                version=version
            )
            self.resources[uri_template] = resource
            logger.debug(f"Registered resource: {uri_template}")
        except Exception as e:
            logger.error(f"Failed to register resource {uri_template}: {e}")
            raise
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            MCPToolNotFoundError: If tool is not found
            MCPInvalidParamsError: If parameters are invalid
            MCPToolExecutionError: If tool execution fails
        """
        self._request_count += 1
        self._last_activity = time.time()
        
        logger.debug(f"Executing tool: {tool_name} with params: {params}")
        
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            self._error_count += 1
            raise MCPToolNotFoundError(tool_name)
            
        tool = self.tools[tool_name]
        
        # Validate parameters against schema
        try:
            self._validate_params(tool.schema, params)
        except MCPValidationError as e:
            self._error_count += 1
            raise MCPInvalidParamsError(str(e), details=e.data)
        
        # Execute tool with performance tracking
        execution_start = time.time()
        try:
            result = tool.func(**params)
            execution_time = time.time() - execution_start
            
            # Track execution time
            if tool_name not in self._tool_execution_times:
                self._tool_execution_times[tool_name] = []
            self._tool_execution_times[tool_name].append(execution_time)
            
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.3f}s")
            return result
            
        except MCPError:
            # Re-raise MCP-specific errors
            self._error_count += 1
            raise
        except Exception as e:
            self._error_count += 1
            logger.error(f"Unhandled error during execution of tool {tool_name}: {e}", exc_info=True)
            raise MCPToolExecutionError(tool_name, e)
    
    def get_resource(self, uri: str) -> Dict[str, Any]:
        """
        Retrieve a resource by URI.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content
            
        Raises:
            MCPResourceNotFoundError: If resource is not found
            MCPToolExecutionError: If resource retrieval fails
        """
        self._request_count += 1
        self._last_activity = time.time()
        
        logger.debug(f"Retrieving resource: {uri}")
        
        # Find matching resource
        for template, resource in self.resources.items():
            if self._match_uri_template(template, uri):
                try:
                    content = resource.retriever(uri=uri)
                    logger.info(f"Resource {uri} retrieved successfully")
                    return content
                except MCPError:
                    # Re-raise MCP-specific errors
                    self._error_count += 1
                    raise
                except Exception as e:
                    self._error_count += 1
                    logger.error(f"Error retrieving resource {uri}: {e}", exc_info=True)
                    raise MCPToolExecutionError(f"resource_retriever_for_{template}", e)
                    
        logger.warning(f"Resource with URI '{uri}' not found")
        self._error_count += 1
        raise MCPResourceNotFoundError(uri)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return the capabilities of this MCP instance.
        
        Returns:
            Dictionary containing server capabilities, tools, and resources
        """
        tools_info = {}
        for name, tool in self.tools.items():
            tools_info[name] = {
                "schema": tool.schema,
                "description": tool.description,
                "module": tool.module,
                "category": tool.category,
                "version": tool.version
            }
            
        resources_info = {}
        for uri_template, resource in self.resources.items():
            resources_info[uri_template] = {
                "description": resource.description,
                "module": resource.module,
                "category": resource.category,
                "version": resource.version
            }
            
        modules_info = {}
        for name, module_info in self.modules.items():
            modules_info[name] = {
                "name": module_info.name,
                "path": str(module_info.path),
                "tools_count": module_info.tools_count,
                "resources_count": module_info.resources_count,
                "status": module_info.status,
                "error_message": module_info.error_message,
                "load_time": module_info.load_time
            }
            
        return {
            "tools": tools_info,
            "resources": resources_info,
            "modules": modules_info,
            "version": "1.0.0",
            "name": "GeneralizedNotationNotation MCP",
            "description": "Model Context Protocol server for GNN (Generalized Notation Notation)",
            "server_info": {
                "uptime": self.uptime,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "last_activity": self._last_activity,
                "sdk_status": _MCP_SDK_STATUS.to_dict()
            }
        }
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get detailed server status information.
        
        Returns:
            Dictionary containing server status details
        """
        # Calculate average execution times
        avg_execution_times = {}
        for tool_name, times in self._tool_execution_times.items():
            if times:
                avg_execution_times[tool_name] = sum(times) / len(times)
        
        return {
            "status": "running",
            "uptime_seconds": self.uptime,
            "uptime_formatted": time.strftime("%H:%M:%S", time.gmtime(self.uptime)),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "modules_count": len(self.modules),
            "modules_loaded": sum(1 for m in self.modules.values() if m.status == "loaded"),
            "modules_failed": sum(1 for m in self.modules.values() if m.status == "error"),
            "last_activity": self._last_activity,
            "avg_execution_times": avg_execution_times,
            "sdk_status": _MCP_SDK_STATUS.to_dict()
        }
    
    def _validate_params(self, schema: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Validate parameters against a JSON schema.
        
        Args:
            schema: JSON schema for validation
            params: Parameters to validate
            
        Raises:
            MCPValidationError: If validation fails
        """
        if not schema or not isinstance(schema, dict):
            return  # No schema to validate against
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Check required parameters
        for param_name in required:
            if param_name not in params:
                raise MCPValidationError(f"Missing required parameter: {param_name}", field=param_name)
        
        # Check parameter types
        for param_name, param_value in params.items():
            if param_name in properties:
                param_schema = properties[param_name]
                expected_type = param_schema.get("type")
                
                if expected_type:
                    type_map = {
                        'string': str,
                        'integer': int,
                        'number': (int, float),
                        'boolean': bool,
                        'array': list,
                        'object': dict
                    }
                    
                    if expected_type in type_map:
                        expected_type_class = type_map[expected_type]
                        if not isinstance(param_value, expected_type_class):
                            raise MCPValidationError(
                                f"Invalid type for parameter '{param_name}'. Expected {expected_type}, got {type(param_value).__name__}.",
                                field=param_name
                            )
    
    def _match_uri_template(self, template: str, uri: str) -> bool:
        """
        Match a URI against a URI template.
        
        Args:
            template: URI template
            uri: URI to match
            
        Returns:
            True if URI matches template
        """
        # Simple template matching - can be enhanced with proper URI template library
        if template == uri:
            return True
        
        # Handle basic template patterns
        if template.endswith('{}') and uri.startswith(template[:-2]):
            return True
        
        if template.endswith('{id}') and uri.startswith(template[:-4]):
            return True
        
        # Add more sophisticated template matching as needed
        return False
    
    @contextmanager
    def _track_performance(self, operation: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.debug(f"{operation} completed in {duration:.3f}s")

# --- Global MCP Instance ---
mcp_instance = MCP()

# --- Initialization Function ---
def initialize(halt_on_missing_sdk: bool = True, force_proceed_flag: bool = False) -> Tuple[MCP, bool, bool]:
    """
    Initialize the MCP by discovering modules and checking SDK status.
    
    Args:
        halt_on_missing_sdk: If True, raises MCPSDKNotFoundError if SDK is missing
        force_proceed_flag: If True, proceeds even if SDK is missing
        
    Returns:
        Tuple of (mcp_instance, sdk_found, all_modules_loaded)
        
    Raises:
        MCPSDKNotFoundError: If SDK is missing and halt_on_missing_sdk is True
    """
    global _critical_mcp_warning_issued
    
    # Check SDK status
    sdk_found = _MCP_SDK_STATUS.check_status()
    
    if not sdk_found:
        if halt_on_missing_sdk and not force_proceed_flag:
            error_message = (
                "MCP SDK is critical for full functionality and was not found or failed to load. "
                "Pipeline is configured to halt. To proceed with limited MCP capabilities, "
                "use a flag like --proceed-without-mcp-sdk or adjust pipeline configuration."
            )
            logger.error(error_message)
            raise MCPSDKNotFoundError(error_message)
        else:
            logger.warning(
                "MCP SDK not found or failed to load, but proceeding with limited functionality."
            )
    
    # Perform module discovery
    all_modules_loaded = mcp_instance.discover_modules()
    
    if all_modules_loaded:
        logger.info("MCP initialization completed successfully")
    else:
        logger.warning("MCP initialization completed with some module loading failures")
    
    return mcp_instance, sdk_found, all_modules_loaded

# --- Utility Functions ---
def get_mcp_instance() -> MCP:
    """Get the global MCP instance, initializing if necessary."""
    if not mcp_instance._modules_discovered:
        initialize()
    return mcp_instance

def list_available_tools() -> List[str]:
    """Get a list of all available tool names."""
    return list(mcp_instance.tools.keys())

def list_available_resources() -> List[str]:
    """Get a list of all available resource URI templates."""
    return list(mcp_instance.resources.keys())

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific tool."""
    if tool_name in mcp_instance.tools:
        tool = mcp_instance.tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.schema,
            "module": tool.module,
            "category": tool.category,
            "version": tool.version
        }
    return None

def get_resource_info(uri_template: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific resource."""
    if uri_template in mcp_instance.resources:
        resource = mcp_instance.resources[uri_template]
        return {
            "uri_template": resource.uri_template,
            "description": resource.description,
            "module": resource.module,
            "category": resource.category,
            "version": resource.version
        }
    return None

# --- Example Usage (for documentation) ---
if __name__ == "__main__":
    # Example of how to use the MCP system
    try:
        # Initialize MCP
        mcp, sdk_found, all_modules_loaded = initialize()
        
        # Get capabilities
        capabilities = mcp.get_capabilities()
        print(f"Available tools: {len(capabilities['tools'])}")
        print(f"Available resources: {len(capabilities['resources'])}")
        
        # Get server status
        status = mcp.get_server_status()
        print(f"Server uptime: {status['uptime_formatted']}")
        
    except Exception as e:
        logger.error(f"Error in MCP initialization: {e}")
        sys.exit(1)

class MCPServer:
    """
    MCP Server implementation for handling JSON-RPC requests.
    
    This class provides a server implementation that can handle
    MCP protocol requests and responses.
    """
    
    def __init__(self, mcp_instance: Optional[MCP] = None):
        """
        Initialize the MCP server.
        
        Args:
            mcp_instance: MCP instance to use for tool execution
        """
        self.mcp = mcp_instance or get_mcp_instance()
        self.running = False
        self.request_handlers = {
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "initialize": self._handle_initialize,
            "notifications/initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "exit": self._handle_exit
        }
    
    def start(self) -> bool:
        """
        Start the MCP server.
        
        Returns:
            True if server started successfully
        """
        try:
            self.running = True
            logger.info("MCP server started")
            return True
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the MCP server.
        
        Returns:
            True if server stopped successfully
        """
        try:
            self.running = False
            logger.info("MCP server stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop MCP server: {e}")
            return False
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP request.
        
        Args:
            request: JSON-RPC request dictionary
        
        Returns:
            JSON-RPC response dictionary
        """
        try:
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method in self.request_handlers:
                result = self.request_handlers[method](params)
                return self._create_success_response(result, request_id)
            else:
                error = MCPError(f"Unknown method: {method}", code=-32601)
                return self._create_error_response(error, request_id)
                
        except Exception as e:
            error = MCPError(f"Internal error: {str(e)}", code=-32603)
            return self._create_error_response(error, request.get("id"))
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.mcp.get_capabilities(),
            "serverInfo": {
                "name": "GNN MCP Server",
                "version": "1.0.0"
            }
        }
    
    def _handle_initialized(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialized notification."""
        return {"status": "initialized"}
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for tool_name, tool in self.mcp.tools.items():
            tools.append({
                "name": tool_name,
                "description": tool.description,
                "inputSchema": tool.schema
            })
        return {"tools": tools}
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        
        if not tool_name:
            raise MCPError("Tool name is required", code=-32602)
        
        result = self.mcp.execute_tool(tool_name, tool_params)
        return {"content": [{"type": "text", "text": str(result)}]}
    
    def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = []
        for uri_template, resource in self.mcp.resources.items():
            resources.append({
                "uri": uri_template,
                "name": resource.description,
                "description": resource.description,
                "mimeType": "application/json"
            })
        return {"resources": resources}
    
    def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if not uri:
            raise MCPError("Resource URI is required", code=-32602)
        
        result = self.mcp.get_resource(uri)
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": str(result)}]}
    
    def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request."""
        self.stop()
        return {"status": "shutdown"}
    
    def _handle_exit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exit request."""
        self.stop()
        return {"status": "exited"}
    
    def _create_success_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
        """Create a successful JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "result": result
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _create_error_response(self, error: MCPError, request_id: Any) -> Dict[str, Any]:
        """Create an error JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": error.code,
                "message": str(error),
                "data": error.data
            }
        }
        if request_id is not None:
            response["id"] = request_id
        return response


def create_mcp_server() -> MCPServer:
    """
    Create an MCP server instance.
    
    Returns:
        MCPServer instance
    """
    return MCPServer()


def start_mcp_server() -> bool:
    """
    Start the global MCP server.
    
    Returns:
        True if server started successfully
    """
    try:
        server = create_mcp_server()
        return server.start()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        return False


def register_tools() -> bool:
    """
    Register all available tools with the MCP server.
    
    Returns:
        True if tools registered successfully
    """
    try:
        mcp = get_mcp_instance()
        return mcp.discover_modules()
    except Exception as e:
        logger.error(f"Failed to register tools: {e}")
        return False