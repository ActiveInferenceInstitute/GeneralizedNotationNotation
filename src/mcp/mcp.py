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
- Performance monitoring and metrics collection
- Advanced module introspection and diagnostics
- Thread-safe operations with proper locking
- Enhanced caching and optimization
- Detailed validation and error reporting
"""

import importlib
import os
import sys
from pathlib import Path
import logging
import inspect
import json
import time
import traceback
import threading
from typing import Dict, List, Any, Callable, Optional, TypedDict, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import weakref
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger("mcp")

# --- Enhanced MCP Exceptions ---
class MCPError(Exception):
    """Base class for MCP related errors."""
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

# --- Enhanced MCP Data Structures ---
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
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    deprecated: bool = False
    experimental: bool = False
    timeout: Optional[float] = None
    max_concurrent: int = 1
    requires_auth: bool = False
    rate_limit: Optional[float] = None
    cache_ttl: Optional[float] = None
    input_validation: bool = True
    output_validation: bool = True
    
    def __post_init__(self):
        """Validate tool configuration after initialization."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.func):
            raise ValueError("Tool function must be callable")
        if not isinstance(self.schema, dict):
            raise ValueError("Tool schema must be a dictionary")
        if not isinstance(self.description, str):
            raise ValueError("Tool description must be a string")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive if specified")
        if self.max_concurrent < 1:
            raise ValueError("Max concurrent must be at least 1")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive if specified")
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be positive if specified")
    
    def get_signature(self) -> str:
        """Get a unique signature for this tool."""
        return hashlib.md5(
            f"{self.name}:{self.module}:{self.version}:{self.schema}".encode()
        ).hexdigest()

@dataclass
class MCPResource:
    """Represents an MCP resource that can be accessed."""
    uri_template: str
    retriever: Callable
    description: str
    module: str = ""
    category: str = ""
    version: str = "1.0.0"
    mime_type: str = "application/json"
    cacheable: bool = True
    tags: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    requires_auth: bool = False
    rate_limit: Optional[float] = None
    cache_ttl: Optional[float] = None
    compression: bool = False
    encryption: bool = False
    
    def __post_init__(self):
        """Validate resource configuration after initialization."""
        if not self.uri_template:
            raise ValueError("Resource URI template cannot be empty")
        if not callable(self.retriever):
            raise ValueError("Resource retriever must be callable")
        if not isinstance(self.description, str):
            raise ValueError("Resource description must be a string")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive if specified")
        if self.rate_limit is not None and self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive if specified")
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            raise ValueError("Cache TTL must be positive if specified")

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
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    file_size: int = 0
    file_hash: str = ""
    import_time: float = 0.0
    register_time: float = 0.0
    memory_usage: Optional[int] = None
    
    def __post_init__(self):
        """Calculate file size and hash if path exists."""
        if self.path.exists():
            try:
                self.file_size = self.path.stat().st_size
                self.file_hash = hashlib.md5(self.path.read_bytes()).hexdigest()
            except Exception as e:
                logger.warning(f"Could not calculate file info for {self.path}: {e}")

@dataclass
class MCPPerformanceMetrics:
    """Enhanced performance metrics for MCP operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    request_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    tool_usage_stats: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    module_load_times: Dict[str, float] = field(default_factory=dict)
    cache_hit_ratio: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: Optional[int] = None
    concurrent_requests: int = 0
    max_concurrent_requests: int = 0
    request_queue_size: int = 0
    max_request_queue_size: int = 0
    
    def update_execution_time(self, execution_time: float):
        """Update execution time statistics."""
        self.total_execution_time += execution_time
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.average_execution_time = self.total_execution_time / self.total_requests
    
    def update_cache_stats(self, hit: bool):
        """Update cache statistics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests > 0:
            self.cache_hit_ratio = self.cache_hits / total_cache_requests

class MCPSDKStatus:
    """Enhanced MCP SDK status tracking."""
    
    def __init__(self):
        self._sdk_found = False
        self._sdk_version = None
        self._sdk_path = None
        self._last_check = 0.0
        self._check_interval = 300.0  # 5 minutes
        self._sdk_capabilities: Dict[str, Any] = {}
        self._sdk_health = "unknown"
        self._sdk_last_health_check = 0.0
    
    def check_status(self) -> bool:
        """Check if MCP SDK is available with enhanced detection."""
        current_time = time.time()
        
        # Cache check results
        if current_time - self._last_check < self._check_interval:
            return self._sdk_found
        
        self._last_check = current_time
        
        try:
            # Check for MCP SDK in various locations
            sdk_paths = [
                Path.home() / ".mcp" / "sdk",
                Path("/usr/local/mcp/sdk"),
                Path("/opt/mcp/sdk"),
                Path.cwd() / "mcp_sdk",
                Path(__file__).parent / "sdk"
            ]
            
            for sdk_path in sdk_paths:
                if sdk_path.exists() and sdk_path.is_dir():
                    self._sdk_found = True
                    self._sdk_path = sdk_path
                    
                    # Try to get version info
                    version_file = sdk_path / "version.txt"
                    if version_file.exists():
                        self._sdk_version = version_file.read_text().strip()
                    
                    # Check SDK health
                    self._check_sdk_health(sdk_path)
                    
                    logger.info(f"MCP SDK found at {sdk_path} (version: {self._sdk_version})")
                    return True
            
            # Also check environment variables
            if os.environ.get("MCP_SDK_PATH"):
                sdk_path = Path(os.environ["MCP_SDK_PATH"])
                if sdk_path.exists() and sdk_path.is_dir():
                    self._sdk_found = True
                    self._sdk_path = sdk_path
                    self._check_sdk_health(sdk_path)
                    logger.info(f"MCP SDK found via environment at {sdk_path}")
                    return True
            
            self._sdk_found = False
            self._sdk_path = None
            self._sdk_version = None
            self._sdk_health = "not_found"
            return False
            
        except Exception as e:
            logger.warning(f"Error checking MCP SDK status: {e}")
            self._sdk_found = False
            self._sdk_health = "error"
            return False
    
    def _check_sdk_health(self, sdk_path: Path):
        """Check the health of the SDK installation."""
        try:
            # Check for essential files
            essential_files = ["mcp.py", "server.py", "client.py"]
            missing_files = [f for f in essential_files if not (sdk_path / f).exists()]
            
            if missing_files:
                self._sdk_health = "incomplete"
                logger.warning(f"MCP SDK missing files: {missing_files}")
            else:
                self._sdk_health = "healthy"
                
            # Check SDK capabilities
            capabilities_file = sdk_path / "capabilities.json"
            if capabilities_file.exists():
                try:
                    self._sdk_capabilities = json.loads(capabilities_file.read_text())
                except Exception as e:
                    logger.warning(f"Could not parse SDK capabilities: {e}")
                    
        except Exception as e:
            logger.warning(f"Error checking SDK health: {e}")
            self._sdk_health = "error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "sdk_found": self._sdk_found,
            "sdk_version": self._sdk_version,
            "sdk_path": str(self._sdk_path) if self._sdk_path else None,
            "sdk_health": self._sdk_health,
            "sdk_capabilities": self._sdk_capabilities,
            "last_check": self._last_check
        }

# Global SDK status instance
_MCP_SDK_STATUS = MCPSDKStatus()

# --- Enhanced Main MCP Class ---
class MCP:
    """
    Enhanced Model Context Protocol implementation.
    
    This class provides the core functionality for:
    - Discovering and loading MCP modules with caching
    - Registering tools and resources with enhanced metadata
    - Executing tools and retrieving resources with performance tracking
    - Managing server capabilities and status
    - Performance monitoring and metrics collection
    - Thread-safe operations with proper locking
    - Enhanced error handling and validation
    """
    
    def __init__(self):
        """Initialize the enhanced MCP server."""
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.modules: Dict[str, MCPModuleInfo] = {}
        self._modules_discovered = False
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._lock = threading.RLock()
        
        # Enhanced performance tracking
        self._performance_metrics = MCPPerformanceMetrics()
        self._tool_execution_times: Dict[str, List[float]] = defaultdict(list)
        self._last_activity = time.time()
        
        # Enhanced module discovery cache
        self._discovery_cache: Dict[str, Any] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        self._cache_lock = threading.Lock()
        
        # Tool execution tracking
        self._active_executions: Dict[str, int] = defaultdict(int)
        self._execution_lock = threading.Lock()
        
        # Rate limiting
        self._rate_limit_timestamps: Dict[str, List[float]] = defaultdict(list)
        self._rate_limit_lock = threading.Lock()
        
        # Caching
        self._result_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()
        
        # Thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="MCP")
        
        logger.info("Enhanced MCP server initialized")
    
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
    
    @property
    def performance_metrics(self) -> MCPPerformanceMetrics:
        """Get performance metrics."""
        return self._performance_metrics
    
    def discover_modules(self, force_refresh: bool = False) -> bool:
        """
        Enhanced module discovery with caching and thread safety.
        
        This method scans the src/ directory for modules with mcp.py files
        and loads them to register their tools and resources with improved
        caching, thread safety, and error handling.
        
        Args:
            force_refresh: If True, force refresh of module discovery cache
            
        Returns:
            bool: True if all modules loaded successfully, False otherwise.
        """
        with self._cache_lock:
            if self._modules_discovered and not force_refresh:
                logger.debug("MCP modules already discovered. Skipping redundant discovery.")
                return True

        with self._lock:
            root_dir = Path(__file__).parent.parent
            logger.info(f"Discovering MCP modules in {root_dir}")
            all_modules_loaded_successfully = True
            
            # Track discovery performance
            discovery_start = time.time()
            
            # Clear existing modules if forcing refresh
            if force_refresh:
                self.modules.clear()
                self.tools.clear()
                self.resources.clear()
                self._modules_discovered = False
                self._discovery_cache.clear()
            
            # Get list of directories to scan
            directories = [d for d in root_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('_')]
            
            # Use thread pool for parallel module loading
            module_load_futures = {}
            
            for directory in directories:
                mcp_file = directory / "mcp.py"
                if not mcp_file.exists():
                    logger.debug(f"No MCP module found in {directory}")
                    continue
                
                # Submit module loading to thread pool
                future = self._executor.submit(self._load_module, directory, mcp_file)
                module_load_futures[directory.name] = future
            
            # Wait for all modules to load
            for module_name, future in module_load_futures.items():
                try:
                    success = future.result(timeout=30.0)  # 30 second timeout per module
                    if not success:
                        all_modules_loaded_successfully = False
                except Exception as e:
                    logger.error(f"Failed to load module {module_name}: {e}")
                    all_modules_loaded_successfully = False
                    
                    # Create error module info
                    self.modules[module_name] = MCPModuleInfo(
                        name=f"src.{module_name}.mcp",
                        path=Path(__file__).parent.parent / module_name / "mcp.py",
                        status="error",
                        error_message=str(e),
                        last_updated=time.time()
                    )

            # Special handling for core MCP tools in the mcp directory itself
            mcp_dir = Path(__file__).parent
            logger.debug(f"Discovering core MCP tools in {mcp_dir}")
            
            # Load SymPy MCP integration
            sympy_mcp_file = mcp_dir / "sympy_mcp.py"
            if sympy_mcp_file.exists():
                try:
                    success = self._load_module(mcp_dir, sympy_mcp_file, module_name="sympy_mcp")
                    if not success:
                        all_modules_loaded_successfully = False
                except Exception as e:
                    logger.error(f"Failed to load core MCP module src.mcp.sympy_mcp: {str(e)}")
                    all_modules_loaded_successfully = False
                    
                    self.modules["sympy_mcp"] = MCPModuleInfo(
                        name="src.mcp.sympy_mcp",
                        path=sympy_mcp_file,
                        status="error",
                        error_message=str(e),
                        last_updated=time.time()
                    )

            discovery_time = time.time() - discovery_start
            logger.info(f"Enhanced module discovery completed in {discovery_time:.2f}s: "
                       f"{len(self.modules)} modules, {len(self.tools)} tools, {len(self.resources)} resources")
            
            self._modules_discovered = True
            self._cache_timestamp = time.time()
            
            return all_modules_loaded_successfully
    
    def _load_module(self, directory: Path, mcp_file: Path, module_name: Optional[str] = None) -> bool:
        """
        Load a single MCP module with enhanced error handling and performance tracking.
        
        Args:
            directory: Directory containing the module
            mcp_file: Path to the mcp.py file
            module_name: Optional custom module name
            
        Returns:
            bool: True if module loaded successfully
        """
        if module_name is None:
            module_name = directory.name
            
        module_start = time.time()
        import_start = time.time()
        
        try:
            # Add parent directory to path if needed
            root_dir = Path(__file__).parent.parent
            if str(root_dir.parent) not in sys.path:
                sys.path.append(str(root_dir.parent))
            
            # Import the module
            full_module_name = f"src.{module_name}.mcp"
            module = importlib.import_module(full_module_name)
            import_time = time.time() - import_start
            
            logger.debug(f"Loaded MCP module: {full_module_name} (import: {import_time:.3f}s)")
            
            # Special handling for llm module initialization
            if full_module_name == "src.llm.mcp":
                if hasattr(module, "initialize_llm_module") and callable(module.initialize_llm_module):
                    logger.debug(f"Calling initialize_llm_module for {full_module_name}")
                    module.initialize_llm_module(self)
                else:
                    logger.warning(f"Module {full_module_name} does not have a callable initialize_llm_module function.")

            # Register tools and resources from the module
            register_start = time.time()
            if hasattr(module, "register_tools") and callable(module.register_tools):
                tools_before = len(self.tools)
                resources_before = len(self.resources)
                
                module.register_tools(self)
                
                tools_added = len(self.tools) - tools_before
                resources_added = len(self.resources) - resources_before
                register_time = time.time() - register_start
                
                module_load_time = time.time() - module_start
                
                # Create module info with enhanced metadata
                self.modules[module_name] = MCPModuleInfo(
                    name=full_module_name,
                    path=mcp_file,
                    tools_count=tools_added,
                    resources_count=resources_added,
                    status="loaded",
                    load_time=module_load_time,
                    import_time=import_time,
                    register_time=register_time,
                    version=getattr(module, "__version__", "1.0.0"),
                    description=getattr(module, "__description__", ""),
                    dependencies=getattr(module, "__dependencies__", []),
                    last_updated=time.time()
                )
                
                # Update performance metrics
                self._performance_metrics.module_load_times[module_name] = module_load_time
                
                logger.info(f"Successfully loaded module {module_name}: "
                           f"{tools_added} tools, {resources_added} resources "
                           f"(load: {module_load_time:.3f}s, import: {import_time:.3f}s, register: {register_time:.3f}s)")
                return True
            else:
                logger.warning(f"Module {full_module_name} found but has no register_tools function.")
                self.modules[module_name] = MCPModuleInfo(
                    name=full_module_name,
                    path=mcp_file,
                    status="no_register_function",
                    import_time=import_time,
                    last_updated=time.time()
                )
                return False
                
        except Exception as e:
            module_load_time = time.time() - module_start
            logger.error(f"Failed to load MCP module {full_module_name}: {str(e)}")
            
            self.modules[module_name] = MCPModuleInfo(
                name=full_module_name,
                path=mcp_file,
                status="error",
                error_message=str(e),
                load_time=module_load_time,
                last_updated=time.time()
            )
            return False

    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any], description: str, 
                     module: str = "", category: str = "", version: str = "1.0.0",
                     tags: Optional[List[str]] = None, examples: Optional[List[Dict[str, Any]]] = None,
                     deprecated: bool = False, experimental: bool = False,
                     timeout: Optional[float] = None, max_concurrent: int = 1, requires_auth: bool = False,
                     rate_limit: Optional[float] = None, cache_ttl: Optional[float] = None,
                     input_validation: bool = True, output_validation: bool = True):
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
            deprecated: Whether the tool is deprecated
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
            if name in self.tools:
                logger.warning(f"Tool '{name}' already registered, overwriting")
            
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
                deprecated=deprecated,
                experimental=experimental,
                timeout=timeout,
                max_concurrent=max_concurrent,
                requires_auth=requires_auth,
                rate_limit=rate_limit,
                cache_ttl=cache_ttl,
                input_validation=input_validation,
                output_validation=output_validation
            )
            
            self.tools[name] = tool
            logger.debug(f"Registered tool: {name}")

    def register_resource(self, uri_template: str, retriever: Callable, description: str,
                         module: str = "", category: str = "", version: str = "1.0.0",
                         mime_type: str = "application/json", cacheable: bool = True,
                         tags: Optional[List[str]] = None,
                         timeout: Optional[float] = None, requires_auth: bool = False,
                         rate_limit: Optional[float] = None, cache_ttl: Optional[float] = None,
                         compression: bool = False, encryption: bool = False):
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
            if uri_template in self.resources:
                logger.warning(f"Resource '{uri_template}' already registered, overwriting")
            
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
                encryption=encryption
            )
            
            self.resources[uri_template] = resource
            logger.debug(f"Registered resource: {uri_template}")

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced tool execution with rate limiting, caching, and performance tracking.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Dict containing the tool execution result
            
        Raises:
            MCPToolNotFoundError: If tool is not found
            MCPInvalidParamsError: If parameters are invalid
            MCPToolExecutionError: If tool execution fails
            MCPPerformanceError: If performance thresholds are exceeded
        """
        with self._lock:
            self._request_count += 1
            self._last_activity = time.time()
            
            # Check if tool exists
            if tool_name not in self.tools:
                available_tools = list(self.tools.keys())
                raise MCPToolNotFoundError(tool_name, available_tools)
            
            tool = self.tools[tool_name]
            
            # Check rate limiting
            if tool.rate_limit is not None:
                self._check_rate_limit(tool_name, tool.rate_limit)
            
            # Check concurrent execution limits
            if tool.max_concurrent > 1:
                self._check_concurrent_limit(tool_name, tool.max_concurrent)
            
            # Check cache for cached results
            if tool.cache_ttl is not None:
                cache_key = self._generate_cache_key(tool_name, params)
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for tool {tool_name}")
                    self._performance_metrics.update_cache_stats(True)
                    return cached_result
            
            # Validate parameters if enabled
            if tool.input_validation:
                try:
                    self._validate_params(tool.schema, params)
                except MCPValidationError as e:
                    e.tool_name = tool_name
                    raise
            
            # Execute the tool with timeout and performance tracking
            execution_start = time.time()
            
            try:
                with self._track_performance(f"tool_execution_{tool_name}"):
                    # Track active execution
                    with self._execution_lock:
                        self._active_executions[tool_name] += 1
                        self._performance_metrics.concurrent_requests += 1
                        self._performance_metrics.max_concurrent_requests = max(
                            self._performance_metrics.max_concurrent_requests,
                            self._performance_metrics.concurrent_requests
                        )
                    
                    # Execute with timeout if specified
                    if tool.timeout is not None:
                        future = self._executor.submit(tool.func, **params)
                        result = future.result(timeout=tool.timeout)
                    else:
                        result = tool.func(**params)
                    
                    execution_time = time.time() - execution_start
                    
                    # Check performance thresholds
                    if execution_time > 30.0:  # 30 second threshold
                        logger.warning(f"Tool {tool_name} execution time ({execution_time:.3f}s) exceeded threshold")
                    
                    # Update performance metrics
                    self._performance_metrics.successful_requests += 1
                    self._performance_metrics.update_execution_time(execution_time)
                    self._performance_metrics.tool_usage_stats[tool_name] = \
                        self._performance_metrics.tool_usage_stats.get(tool_name, 0) + 1
                    
                    # Store execution time for this tool
                    self._tool_execution_times[tool_name].append(execution_time)
                    
                    # Cache result if TTL is specified
                    if tool.cache_ttl is not None:
                        cache_key = self._generate_cache_key(tool_name, params)
                        self._cache_result(cache_key, result, tool.cache_ttl)
                    
                    # Validate output if enabled
                    if tool.output_validation:
                        self._validate_output(result)
                    
                    logger.debug(f"Tool {tool_name} executed successfully in {execution_time:.3f}s")
                    return result
                    
            except Exception as e:
                execution_time = time.time() - execution_start
                self._performance_metrics.failed_requests += 1
                self._performance_metrics.error_counts[tool_name] = \
                    self._performance_metrics.error_counts.get(tool_name, 0) + 1
                
                # Log detailed error information
                logger.error(f"Tool {tool_name} execution failed after {execution_time:.3f}s: {e}")
                logger.debug(f"Tool {tool_name} parameters: {params}")
                
                raise MCPToolExecutionError(tool_name, e, execution_time)
                
            finally:
                # Clean up execution tracking
                with self._execution_lock:
                    self._active_executions[tool_name] = max(0, self._active_executions[tool_name] - 1)
                    self._performance_metrics.concurrent_requests = max(0, self._performance_metrics.concurrent_requests - 1)

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
            
            try:
                # Retrieve resource content
                content = matching_resource.retriever(uri)
                
                # Add metadata
                result = {
                    "content": content,
                    "uri": uri,
                    "mime_type": matching_resource.mime_type,
                    "cacheable": matching_resource.cacheable,
                    "retrieved_at": time.time()
                }
                
                logger.debug(f"Resource '{uri}' retrieved successfully")
                return result
                
            except Exception as e:
                logger.error(f"Resource '{uri}' retrieval failed: {e}")
                raise MCPResourceNotFoundError(uri)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities including all available tools and resources."""
        with self._lock:
            tools_list = []
            for tool in self.tools.values():
                tools_list.append({
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.schema,
                    "module": tool.module,
                    "category": tool.category,
                    "version": tool.version,
                    "tags": tool.tags,
                    "examples": tool.examples,
                    "deprecated": tool.deprecated,
                    "experimental": tool.experimental,
                    "timeout": tool.timeout,
                    "max_concurrent": tool.max_concurrent,
                    "requires_auth": tool.requires_auth,
                    "rate_limit": tool.rate_limit,
                    "cache_ttl": tool.cache_ttl,
                    "input_validation": tool.input_validation,
                    "output_validation": tool.output_validation
                })
            
            resources_list = []
            for resource in self.resources.values():
                resources_list.append({
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
                    "encryption": resource.encryption
                })
            
            return {
                "tools": tools_list,
                "resources": resources_list,
                "server": {
                    "name": "GNN MCP Server",
                    "version": "1.0.0",
                    "description": "Model Context Protocol server for GeneralizedNotationNotation",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": True}
                    }
                }
            }

    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status information."""
        with self._lock:
            uptime_seconds = self.uptime
            uptime_str = time.strftime("%H:%M:%S", time.gmtime(uptime_seconds))
            
            # Calculate tool categories
            categories = defaultdict(int)
            for tool in self.tools.values():
                categories[tool.category or "uncategorized"] += 1
            
            # Calculate resource categories
            resource_categories = defaultdict(int)
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
                    "min_execution_time": self._performance_metrics.min_execution_time if self._performance_metrics.min_execution_time != float('inf') else 0.0
                }
            }

    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific module."""
        with self._lock:
            if module_name not in self.modules:
                return None
            
            module_info = self.modules[module_name]
            
            # Get tools for this module
            module_tools = [
                tool.name for tool in self.tools.values()
                if tool.module == module_name
            ]
            
            # Get resources for this module
            module_resources = [
                resource.uri_template for resource in self.resources.values()
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
                "resources": module_resources
            }

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tool."""
        with self._lock:
            if tool_name not in self.tools:
                return None
            
            tool = self.tools[tool_name]
            
            # Get execution statistics
            execution_times = self._tool_execution_times.get(tool_name, [])
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            return {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
                "module": tool.module,
                "category": tool.category,
                "version": tool.version,
                "tags": tool.tags,
                "examples": tool.examples,
                "deprecated": tool.deprecated,
                "experimental": tool.experimental,
                "usage_count": self._performance_metrics.tool_usage_stats.get(tool_name, 0),
                "average_execution_time": avg_execution_time,
                "execution_count": len(execution_times),
                "timeout": tool.timeout,
                "max_concurrent": tool.max_concurrent,
                "requires_auth": tool.requires_auth,
                "rate_limit": tool.rate_limit,
                "cache_ttl": tool.cache_ttl,
                "input_validation": tool.input_validation,
                "output_validation": tool.output_validation
            }

    def _validate_params(self, schema: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Enhanced parameter validation against schema with detailed error reporting.
        
        Args:
            schema: JSON schema for validation
            params: Parameters to validate
            
        Raises:
            MCPValidationError: If validation fails
        """
        if not isinstance(params, dict):
            raise MCPValidationError("Parameters must be a dictionary")
        
        # Check required fields
        if "required" in schema:
            for required_field in schema["required"]:
                if required_field not in params:
                    raise MCPValidationError(
                        f"Required parameter '{required_field}' is missing",
                        field=required_field
                    )
        
        # Validate properties
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                if field_name in params:
                    field_value = params[field_name]
                    self._validate_field(field_name, field_value, field_schema)
        
        # Validate additional constraints
        if "minProperties" in schema and len(params) < schema["minProperties"]:
            raise MCPValidationError(
                f"Too few properties: {len(params)} < {schema['minProperties']}"
            )
        
        if "maxProperties" in schema and len(params) > schema["maxProperties"]:
            raise MCPValidationError(
                f"Too many properties: {len(params)} > {schema['maxProperties']}"
            )
    
    def _validate_field(self, field_name: str, field_value: Any, field_schema: Dict[str, Any]) -> None:
        """
        Validate a single field against its schema.
        
        Args:
            field_name: Name of the field
            field_value: Value to validate
            field_schema: Schema for the field
            
        Raises:
            MCPValidationError: If validation fails
        """
        field_type = field_schema.get("type")
        
        # Type validation
        if field_type == "string":
            if not isinstance(field_value, str):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be a string",
                    field=field_name,
                    value=field_value
                )
            
            # String-specific validations
            if "minLength" in field_schema and len(field_value) < field_schema["minLength"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too short: {len(field_value)} < {field_schema['minLength']}",
                    field=field_name,
                    value=field_value
                )
            
            if "maxLength" in field_schema and len(field_value) > field_schema["maxLength"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too long: {len(field_value)} > {field_schema['maxLength']}",
                    field=field_name,
                    value=field_value
                )
            
            if "pattern" in field_schema:
                import re
                if not re.match(field_schema["pattern"], field_value):
                    raise MCPValidationError(
                        f"Parameter '{field_name}' does not match pattern: {field_schema['pattern']}",
                        field=field_name,
                        value=field_value
                    )
            
            if "enum" in field_schema and field_value not in field_schema["enum"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be one of: {field_schema['enum']}",
                    field=field_name,
                    value=field_value
                )
        
        elif field_type == "integer":
            if not isinstance(field_value, int):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be an integer",
                    field=field_name,
                    value=field_value
                )
            
            # Integer-specific validations
            if "minimum" in field_schema and field_value < field_schema["minimum"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too small: {field_value} < {field_schema['minimum']}",
                    field=field_name,
                    value=field_value
                )
            
            if "maximum" in field_schema and field_value > field_schema["maximum"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too large: {field_value} > {field_schema['maximum']}",
                    field=field_name,
                    value=field_value
                )
        
        elif field_type == "number":
            if not isinstance(field_value, (int, float)):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be a number",
                    field=field_name,
                    value=field_value
                )
            
            # Number-specific validations
            if "minimum" in field_schema and field_value < field_schema["minimum"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too small: {field_value} < {field_schema['minimum']}",
                    field=field_name,
                    value=field_value
                )
            
            if "maximum" in field_schema and field_value > field_schema["maximum"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too large: {field_value} > {field_schema['maximum']}",
                    field=field_name,
                    value=field_value
                )
        
        elif field_type == "boolean":
            if not isinstance(field_value, bool):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be a boolean",
                    field=field_name,
                    value=field_value
                )
        
        elif field_type == "array":
            if not isinstance(field_value, list):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be an array",
                    field=field_name,
                    value=field_value
                )
            
            # Array-specific validations
            if "minItems" in field_schema and len(field_value) < field_schema["minItems"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too few items: {len(field_value)} < {field_schema['minItems']}",
                    field=field_name,
                    value=field_value
                )
            
            if "maxItems" in field_schema and len(field_value) > field_schema["maxItems"]:
                raise MCPValidationError(
                    f"Parameter '{field_name}' too many items: {len(field_value)} > {field_schema['maxItems']}",
                    field=field_name,
                    value=field_value
                )
            
            # Validate array items if schema provided
            if "items" in field_schema:
                for i, item in enumerate(field_value):
                    try:
                        self._validate_field(f"{field_name}[{i}]", item, field_schema["items"])
                    except MCPValidationError as e:
                        raise MCPValidationError(
                            f"Array item validation failed: {e}",
                            field=field_name,
                            value=field_value
                        )
        
        elif field_type == "object":
            if not isinstance(field_value, dict):
                raise MCPValidationError(
                    f"Parameter '{field_name}' must be an object",
                    field=field_name,
                    value=field_value
                )
            
            # Object-specific validations
            if "properties" in field_schema:
                for prop_name, prop_value in field_value.items():
                    if prop_name in field_schema["properties"]:
                        try:
                            self._validate_field(f"{field_name}.{prop_name}", prop_value, field_schema["properties"][prop_name])
                        except MCPValidationError as e:
                            raise MCPValidationError(
                                f"Object property validation failed: {e}",
                                field=field_name,
                                value=field_value
                            )
            
            # Check for additional properties
            if "additionalProperties" in field_schema and field_schema["additionalProperties"] is False:
                allowed_props = set(field_schema.get("properties", {}).keys())
                actual_props = set(field_value.keys())
                extra_props = actual_props - allowed_props
                if extra_props:
                    raise MCPValidationError(
                        f"Parameter '{field_name}' has additional properties not allowed: {extra_props}",
                        field=field_name,
                        value=field_value
                    )

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
                    continue  # Parameter placeholder
                if template_part != uri_part:
                    return False
            
            return True
        
        return False

    @contextmanager
    def _track_performance(self, operation: str):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            logger.debug(f"Operation '{operation}' completed in {execution_time:.4f}s")

    def _check_rate_limit(self, tool_name: str, rate_limit: float):
        """Check if tool execution is within rate limits."""
        with self._rate_limit_lock:
            current_time = time.time()
            timestamps = self._rate_limit_timestamps[tool_name]
            
            # Remove old timestamps (older than 1 second)
            timestamps[:] = [ts for ts in timestamps if current_time - ts < 1.0]
            
            # Check if we're within rate limit
            if len(timestamps) >= rate_limit:
                raise MCPValidationError(
                    f"Rate limit exceeded for tool {tool_name}: {len(timestamps)} requests in 1 second",
                    tool_name=tool_name
                )
            
            # Add current timestamp
            timestamps.append(current_time)

    def _check_concurrent_limit(self, tool_name: str, max_concurrent: int):
        """Check if tool execution is within concurrent limits."""
        with self._execution_lock:
            current_concurrent = self._active_executions[tool_name]
            if current_concurrent >= max_concurrent:
                raise MCPValidationError(
                    f"Concurrent execution limit exceeded for tool {tool_name}: {current_concurrent}/{max_concurrent}",
                    tool_name=tool_name
                )

    def _generate_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for tool execution."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{tool_name}:{param_str}".encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get a cached result if available and not expired."""
        with self._cache_lock:
            if cache_key in self._result_cache:
                result, timestamp = self._result_cache[cache_key]
                if time.time() - timestamp < 300:  # 5 minute default TTL
                    return result
                else:
                    # Remove expired cache entry
                    del self._result_cache[cache_key]
            return None

    def _cache_result(self, cache_key: str, result: Any, ttl: float):
        """Cache a tool execution result."""
        with self._cache_lock:
            self._result_cache[cache_key] = (result, time.time())
            
            # Limit cache size
            if len(self._result_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._result_cache.keys(),
                    key=lambda k: self._result_cache[k][1]
                )[:100]
                for key in oldest_keys:
                    del self._result_cache[key]

    def _validate_output(self, result: Any):
        """Validate tool output (basic validation)."""
        if result is None:
            raise MCPValidationError("Tool output cannot be None")
        
        # Add more validation as needed
        if isinstance(result, dict) and "error" in result:
            raise MCPValidationError(f"Tool returned error: {result['error']}")

    def get_enhanced_server_status(self) -> Dict[str, Any]:
        """
        Get enhanced server status with detailed metrics and health information.
        
        Returns:
            Dict containing comprehensive server status
        """
        with self._lock:
            # Get memory usage if available
            memory_usage = None
            try:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss
            except ImportError:
                pass
            
            # Calculate cache statistics
            cache_size = len(self._result_cache)
            cache_memory_estimate = cache_size * 1024  # Rough estimate
            
            # Get active executions
            active_executions = dict(self._active_executions)
            
            # Get rate limit status
            rate_limit_status = {}
            with self._rate_limit_lock:
                for tool_name, timestamps in self._rate_limit_timestamps.items():
                    current_time = time.time()
                    recent_requests = len([ts for ts in timestamps if current_time - ts < 1.0])
                    rate_limit_status[tool_name] = {
                        "recent_requests": recent_requests,
                        "total_requests": len(timestamps)
                    }
            
            return {
                "server_info": {
                    "name": "GNN MCP Server",
                    "version": "2.0.0",
                    "uptime": self.uptime,
                    "start_time": self._start_time,
                    "last_activity": self._last_activity
                },
                "performance": {
                    "total_requests": self._performance_metrics.total_requests,
                    "successful_requests": self._performance_metrics.successful_requests,
                    "failed_requests": self._performance_metrics.failed_requests,
                    "success_rate": (
                        self._performance_metrics.successful_requests / max(1, self._performance_metrics.total_requests)
                    ),
                    "average_execution_time": self._performance_metrics.average_execution_time,
                    "max_execution_time": self._performance_metrics.max_execution_time,
                    "min_execution_time": self._performance_metrics.min_execution_time if self._performance_metrics.min_execution_time != float('inf') else 0.0,
                    "cache_hit_ratio": self._performance_metrics.cache_hit_ratio,
                    "cache_hits": self._performance_metrics.cache_hits,
                    "cache_misses": self._performance_metrics.cache_misses,
                    "concurrent_requests": self._performance_metrics.concurrent_requests,
                    "max_concurrent_requests": self._performance_metrics.max_concurrent_requests
                },
                "resources": {
                    "tools_count": len(self.tools),
                    "resources_count": len(self.resources),
                    "modules_count": len(self.modules),
                    "memory_usage_bytes": memory_usage,
                    "cache_size": cache_size,
                    "cache_memory_estimate_bytes": cache_memory_estimate
                },
                "modules": {
                    module_name: {
                        "status": info.status,
                        "tools_count": info.tools_count,
                        "resources_count": info.resources_count,
                        "load_time": info.load_time,
                        "last_updated": info.last_updated,
                        "error_message": info.error_message
                    }
                    for module_name, info in self.modules.items()
                },
                "active_executions": active_executions,
                "rate_limit_status": rate_limit_status,
                "sdk_status": _MCP_SDK_STATUS.to_dict(),
                "health": {
                    "status": "healthy" if self._performance_metrics.failed_requests / max(1, self._performance_metrics.total_requests) < 0.1 else "degraded",
                    "error_rate": self._performance_metrics.failed_requests / max(1, self._performance_metrics.total_requests),
                    "cache_efficiency": self._performance_metrics.cache_hit_ratio,
                    "concurrent_load": self._performance_metrics.concurrent_requests / 10.0  # Normalized to max workers
                }
            }

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear all caches and return statistics.
        
        Returns:
            Dict containing cache clearing statistics
        """
        with self._cache_lock:
            cache_size_before = len(self._result_cache)
            self._result_cache.clear()
            
            discovery_cache_size = len(self._discovery_cache)
            self._discovery_cache.clear()
            
            return {
                "result_cache_cleared": cache_size_before,
                "discovery_cache_cleared": discovery_cache_size,
                "timestamp": time.time()
            }

    def get_tool_performance_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed performance statistics for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dict containing tool performance statistics or None if tool not found
        """
        if tool_name not in self.tools:
            return None
        
        execution_times = self._tool_execution_times.get(tool_name, [])
        if not execution_times:
            return {
                "tool_name": tool_name,
                "execution_count": 0,
                "average_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0,
                "total_execution_time": 0.0,
                "error_count": self._performance_metrics.error_counts.get(tool_name, 0),
                "success_rate": 1.0
            }
        
        total_executions = len(execution_times)
        total_time = sum(execution_times)
        avg_time = total_time / total_executions
        min_time = min(execution_times)
        max_time = max(execution_times)
        error_count = self._performance_metrics.error_counts.get(tool_name, 0)
        success_count = self._performance_metrics.tool_usage_stats.get(tool_name, 0)
        total_attempts = success_count + error_count
        success_rate = success_count / max(1, total_attempts)
        
        return {
            "tool_name": tool_name,
            "execution_count": total_executions,
            "average_execution_time": avg_time,
            "min_execution_time": min_time,
            "max_execution_time": max_time,
            "total_execution_time": total_time,
            "error_count": error_count,
            "success_count": success_count,
            "success_rate": success_rate,
            "recent_executions": execution_times[-10:] if len(execution_times) > 10 else execution_times
        }

    def shutdown(self) -> Dict[str, Any]:
        """
        Gracefully shutdown the MCP server.
        
        Returns:
            Dict containing shutdown statistics
        """
        logger.info("Shutting down MCP server...")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Clear caches
        cache_stats = self.clear_cache()
        
        # Get final statistics
        final_stats = {
            "uptime": self.uptime,
            "total_requests": self._performance_metrics.total_requests,
            "successful_requests": self._performance_metrics.successful_requests,
            "failed_requests": self._performance_metrics.failed_requests,
            "cache_stats": cache_stats,
            "shutdown_time": time.time()
        }
        
        logger.info(f"MCP server shutdown complete: {final_stats}")
        return final_stats

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

def get_mcp_instance() -> MCP:
    """Get the global MCP instance."""
    return mcp_instance

def list_available_tools() -> List[str]:
    """List all available tool names."""
    return list(mcp_instance.tools.keys())

def list_available_resources() -> List[str]:
    """List all available resource URI templates."""
    return list(mcp_instance.resources.keys())

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific tool."""
    return mcp_instance.get_tool_info(tool_name)

def get_resource_info(uri_template: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific resource."""
    if uri_template in mcp_instance.resources:
        resource = mcp_instance.resources[uri_template]
        return {
            "uri_template": resource.uri_template,
            "description": resource.description,
            "module": resource.module,
            "category": resource.category,
            "version": resource.version,
            "mime_type": resource.mime_type,
            "cacheable": resource.cacheable,
            "tags": resource.tags
        }
    return None

# --- MCP Server Implementation ---
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
        """Start the MCP server."""
        if self.running:
            logger.warning("MCP server is already running")
            return False
        
        self.running = True
        logger.info("MCP server started")
        return True
    
    def stop(self) -> bool:
        """Stop the MCP server."""
        if not self.running:
            logger.warning("MCP server is not running")
            return False
        
        self.running = False
        logger.info("MCP server stopped")
        return True
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming JSON-RPC request.
        
        Args:
            request: JSON-RPC request dictionary
            
        Returns:
            JSON-RPC response dictionary
        """
        try:
            # Validate request structure
            if not isinstance(request, dict):
                return self._create_error_response(-32700, "Parse error", "Invalid JSON")
            
            if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
                return self._create_error_response(-32600, "Invalid Request", "Missing or invalid jsonrpc field")
            
            if "method" not in request:
                return self._create_error_response(-32600, "Invalid Request", "Missing method field")
            
            method = request["method"]
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Handle the request
            if method in self.request_handlers:
                result = self.request_handlers[method](params)
                return self._create_success_response(result, request_id)
            else:
                # Try to execute as a direct tool call
                if method in self.mcp.tools:
                    result = self.mcp.execute_tool(method, params)
                    return self._create_success_response(result, request_id)
                else:
                    return self._create_error_response(-32601, "Method not found", f"Method '{method}' not found", request_id)
                    
        except MCPError as e:
            return self._create_error_response(e.code, type(e).__name__, str(e), request.get("id"))
        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}")
            return self._create_error_response(-32603, "Internal error", str(e), request.get("id"))
    
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
        return {}
    
    def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {"tools": self.mcp.get_capabilities()["tools"]}
    
    def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        
        if not tool_name:
            raise MCPInvalidParamsError("Tool name is required")
        
        result = self.mcp.execute_tool(tool_name, tool_params)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {"resources": self.mcp.get_capabilities()["resources"]}
    
    def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if not uri:
            raise MCPInvalidParamsError("Resource URI is required")
        
        result = self.mcp.get_resource(uri)
        return {"contents": [{"uri": uri, "mimeType": result["mime_type"], "text": json.dumps(result["content"])}]}
    
    def _handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle shutdown request."""
        self.stop()
        return {}
    
    def _handle_exit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exit request."""
        self.stop()
        return {}
    
    def _create_success_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
        """Create a successful JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "result": result
        }
        if request_id is not None:
            response["id"] = request_id
        return response
    
    def _create_error_response(self, code: int, message: str, data: Any = None, request_id: Any = None) -> Dict[str, Any]:
        """Create an error JSON-RPC response."""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if data is not None:
            response["error"]["data"] = data
        if request_id is not None:
            response["id"] = request_id
        return response

def create_mcp_server(mcp_instance: Optional[MCP] = None) -> MCPServer:
    """Create a new MCP server instance."""
    return MCPServer(mcp_instance)

def start_mcp_server(mcp_instance: Optional[MCP] = None) -> bool:
    """Start the MCP server."""
    server = create_mcp_server(mcp_instance)
    return server.start()

def register_tools() -> bool:
    """Register tools with the global MCP instance."""
    try:
        # This function is called by other modules to register their tools
        # The actual registration happens in the module's mcp.py file
        return True
    except Exception as e:
        logger.error(f"Failed to register tools: {e}")
        return False

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