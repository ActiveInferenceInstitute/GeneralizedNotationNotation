# Model Context Protocol (MCP) Implementation Documentation

## Overview

The Model Context Protocol (MCP) implementation for the GeneralizedNotationNotation (GNN) project provides a server with features for tool discovery, registration, execution, and monitoring. This implementation extends the standard MCP specification with error handling, performance monitoring, caching, rate limiting, and thread safety.

## Key Features

### Core Features
- **JSON-RPC 2.0 Compliance**: Compliance with the MCP specification
- **Dynamic Module Discovery**: Automatic discovery and loading of MCP modules
- **Tool and Resource Registration**: Tool and resource management
- **Multiple Transport Layers**: Support for stdio and HTTP transport
- **CLI Interface**: Command-line access to MCP functionality

### Features
- **Advanced Error Handling**: Detailed error reporting with context information
- **Performance Monitoring**: Comprehensive metrics and health monitoring
- **Caching System**: Intelligent result caching with TTL support
- **Rate Limiting**: Configurable rate limiting per tool
- **Concurrent Execution Control**: Limits on concurrent tool executions
- **Thread Safety**: Full thread-safe operations with proper locking
- **Enhanced Validation**: Comprehensive parameter validation with detailed error messages
- **Health Monitoring**: Real-time health status and diagnostics
- **Memory Management**: Efficient memory usage and cleanup

## Architecture

### Core Components

```
src/mcp/
├── mcp.py                 # Enhanced core MCP server implementation
├── server_stdio.py        # stdio transport server
├── server_http.py         # HTTP transport server
├── cli.py                 # Command-line interface
├── meta_mcp.py           # Meta-tools for server introspection
├── sympy_mcp.py          # SymPy integration tools
├── sympy_mcp_client.py   # SymPy MCP client
├── npx_inspector.py      # NPX inspector utilities
├── README.md             # Basic documentation
├── COMPREHENSIVE_DOCUMENTATION.md  # This comprehensive documentation
└── *.md                  # Additional documentation files
```

### Enhanced Data Structures

#### MCPTool
Enhanced tool representation with additional metadata:

```python
@dataclass
class MCPTool:
    name: str                    # Tool name
    func: Callable              # Tool function
    schema: Dict[str, Any]      # JSON schema for parameters
    description: str            # Tool description
    module: str = ""            # Source module
    category: str = ""          # Tool category
    version: str = "1.0.0"      # Tool version
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    examples: List[Dict[str, Any]] = field(default_factory=list)  # Usage examples
    deprecated: bool = False    # Deprecation flag
    experimental: bool = False  # Experimental flag
    timeout: Optional[float] = None  # Execution timeout
    max_concurrent: int = 1     # Max concurrent executions
    requires_auth: bool = False # Authentication requirement
    rate_limit: Optional[float] = None  # Rate limit (requests/second)
    cache_ttl: Optional[float] = None  # Cache TTL
    input_validation: bool = True   # Enable input validation
    output_validation: bool = True  # Enable output validation
```

#### MCPResource
Enhanced resource representation:

```python
@dataclass
class MCPResource:
    uri_template: str           # URI template
    retriever: Callable         # Resource retriever function
    description: str            # Resource description
    module: str = ""            # Source module
    category: str = ""          # Resource category
    version: str = "1.0.0"      # Resource version
    mime_type: str = "application/json"  # MIME type
    cacheable: bool = True      # Cacheable flag
    tags: List[str] = field(default_factory=list)  # Tags
    timeout: Optional[float] = None  # Retrieval timeout
    requires_auth: bool = False # Authentication requirement
    rate_limit: Optional[float] = None  # Rate limit
    cache_ttl: Optional[float] = None  # Cache TTL
    compression: bool = False   # Compression flag
    encryption: bool = False    # Encryption flag
```

### Enhanced Error Classes

The implementation provides comprehensive error handling with detailed context:

```python
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

class MCPToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""
    
class MCPResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""
    
class MCPInvalidParamsError(MCPError):
    """Raised when tool parameters are invalid."""
    
class MCPToolExecutionError(MCPError):
    """Raised when tool execution fails."""
    
class MCPSDKNotFoundError(MCPError):
    """Raised when required SDK is not found."""
    
class MCPValidationError(MCPError):
    """Raised when validation fails."""
    
class MCPModuleLoadError(MCPError):
    """Raised when a module fails to load."""
    
class MCPPerformanceError(MCPError):
    """Raised when performance thresholds are exceeded."""
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required dependencies (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd GeneralizedNotationNotation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m src.mcp.cli --help
```

### Basic Usage

#### Starting the MCP Server

**stdio Transport (Recommended for local use):**
```bash
python -m src.mcp.cli server --transport stdio
```

**HTTP Transport (For network access):**
```bash
python -m src.mcp.cli server --transport http --host 0.0.0.0 --port 8080
```

#### Using the CLI

**List all available tools:**
```bash
python -m src.mcp.cli list --format human
```

**Execute a tool:**
```bash
python -m src.mcp.cli execute get_gnn_files --params '{"target_dir": "doc", "recursive": true}'
```

**Get server status:**
```bash
python -m src.mcp.cli status --format json
```

## API Reference

### Core MCP Class

#### Initialization
```python
from mcp import MCP

# Create MCP instance
mcp_instance = MCP()

# Initialize with module discovery
mcp_instance.discover_modules()
```

#### Tool Registration
```python
def my_tool(param1: str, param2: int) -> Dict[str, Any]:
    return {"result": f"{param1}_{param2}", "success": True}

schema = {
    "type": "object",
    "properties": {
        "param1": {"type": "string", "minLength": 1},
        "param2": {"type": "integer", "minimum": 0}
    },
    "required": ["param1", "param2"]
}

mcp_instance.register_tool(
    name="my_tool",
    func=my_tool,
    schema=schema,
    description="My custom tool",
    module="my_module",
    category="custom",
    version="1.0.0",
    tags=["custom", "example"],
    examples=[{"param1": "hello", "param2": 42}],
    timeout=30.0,
    max_concurrent=5,
    rate_limit=10.0,
    cache_ttl=300.0,
    input_validation=True,
    output_validation=True
)
```

#### Resource Registration
```python
def my_resource_retriever(uri: str) -> Dict[str, Any]:
    return {"content": f"Resource content for {uri}", "uri": uri}

mcp_instance.register_resource(
    uri_template="my://{resource_id}",
    retriever=my_resource_retriever,
    description="My custom resource",
    module="my_module",
    category="custom",
    version="1.0.0",
    mime_type="application/json",
    cacheable=True,
    tags=["custom", "resource"],
    timeout=30.0,
    rate_limit=10.0,
    cache_ttl=300.0
)
```

#### Tool Execution
```python
# Execute a tool
result = mcp_instance.execute_tool("my_tool", {
    "param1": "hello",
    "param2": 42
})

# Get tool performance statistics
stats = mcp_instance.get_tool_performance_stats("my_tool")
```

#### Resource Access
```python
# Get a resource
result = mcp_instance.get_resource("my://my_resource")
```

#### Server Status and Monitoring
```python
# Get enhanced server status
status = mcp_instance.get_enhanced_server_status()

# Get basic server status
basic_status = mcp_instance.get_server_status()

# Get capabilities
capabilities = mcp_instance.get_capabilities()

# Clear caches
cache_stats = mcp_instance.clear_cache()
```

### Enhanced Parameter Validation

The implementation provides comprehensive parameter validation:

```python
# Complex schema with validation
schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 3,
            "maxLength": 50,
            "pattern": "^[a-zA-Z0-9_]+$"
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150
        },
        "tags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {"type": "string"}
        },
        "settings": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "timeout": {"type": "number", "minimum": 0}
            },
            "required": ["enabled"]
        }
    },
    "required": ["name", "age"],
    "additionalProperties": False
}
```

### Caching System

The implementation includes intelligent caching:

```python
# Register tool with caching
mcp_instance.register_tool(
    name="expensive_operation",
    func=expensive_function,
    schema=schema,
    description="An expensive operation with caching",
    cache_ttl=3600.0  # Cache for 1 hour
)

# Execute (will be cached)
result1 = mcp_instance.execute_tool("expensive_operation", {"param": "value"})

# Execute again (will use cache)
result2 = mcp_instance.execute_tool("expensive_operation", {"param": "value"})

# Clear cache
cache_stats = mcp_instance.clear_cache()
```

### Rate Limiting

Configure rate limiting per tool:

```python
# Register tool with rate limiting
mcp_instance.register_tool(
    name="rate_limited_tool",
    func=my_function,
    schema=schema,
    description="A rate-limited tool",
    rate_limit=5.0  # 5 requests per second
)

# Execute (will be rate limited if exceeded)
result = mcp_instance.execute_tool("rate_limited_tool", {})
```

### Concurrent Execution Control

Limit concurrent executions:

```python
# Register tool with concurrent limits
mcp_instance.register_tool(
    name="concurrent_limited_tool",
    func=my_function,
    schema=schema,
    description="A tool with concurrent execution limits",
    max_concurrent=3  # Only 3 concurrent executions allowed
)
```

## Transport Layers

### stdio Server

The stdio server provides local process communication:

```python
from mcp.server_stdio import StdioServer

# Create and start stdio server
server = StdioServer()
server.start()
```

**Usage:**
```bash
python -m src.mcp.cli server --transport stdio
```

### HTTP Server

The HTTP server provides network-based access:

```python
from mcp.server_http import MCPHTTPHandler
from http.server import HTTPServer

# Create HTTP server
server = HTTPServer(('localhost', 8080), MCPHTTPHandler)
server.serve_forever()
```

**Usage:**
```bash
python -m src.mcp.cli server --transport http --host 0.0.0.0 --port 8080
```

## JSON-RPC API

### Standard MCP Methods

#### Get Capabilities
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "mcp.capabilities",
  "params": {}
}
```

#### Execute Tool
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "mcp.tool.execute",
  "params": {
    "name": "my_tool",
    "params": {
      "param1": "hello",
      "param2": 42
    }
  }
}
```

#### Get Resource
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "mcp.resource.get",
  "params": {
    "uri": "my://my_resource"
  }
}
```

### Direct Tool Invocation

Tools can also be invoked directly:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "my_tool",
  "params": {
    "param1": "hello",
    "param2": 42
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Parameter 'param1' must be a string",
    "data": {
      "field": "param1",
      "value": 123,
      "tool_name": "my_tool",
      "module_name": "my_module"
    }
  }
}
```

### Error Codes

- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
- `-32000`: MCP-specific errors
- `-32001`: SDK not found
- `-32002`: Resource retrieval errors
- `-32003`: Module loading errors
- `-32004`: Performance errors

## Performance Monitoring

### Performance Metrics

The implementation tracks comprehensive performance metrics:

```python
# Get performance metrics
status = mcp_instance.get_enhanced_server_status()
performance = status["performance"]

# Available metrics
print(f"Total requests: {performance['total_requests']}")
print(f"Success rate: {performance['success_rate']:.2%}")
print(f"Average execution time: {performance['average_execution_time']:.3f}s")
print(f"Cache hit ratio: {performance['cache_hit_ratio']:.2%}")
print(f"Concurrent requests: {performance['concurrent_requests']}")
```

### Health Monitoring

Real-time health status:

```python
# Get health status
status = mcp_instance.get_enhanced_server_status()
health = status["health"]

print(f"Status: {health['status']}")  # healthy, degraded, error
print(f"Error rate: {health['error_rate']:.2%}")
print(f"Cache efficiency: {health['cache_efficiency']:.2%}")
print(f"Concurrent load: {health['concurrent_load']:.2%}")
```

### Tool Performance Statistics

Get detailed statistics for specific tools:

```python
# Get tool performance stats
stats = mcp_instance.get_tool_performance_stats("my_tool")

if stats:
    print(f"Execution count: {stats['execution_count']}")
    print(f"Average execution time: {stats['average_execution_time']:.3f}s")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Error count: {stats['error_count']}")
```

## Module Development

### Creating MCP Modules

Create a new MCP module by adding an `mcp.py` file to your module directory:

```python
# src/my_module/mcp.py

def register_tools(mcp_instance):
    """Register tools for this module."""
    
    # Define tool function
    def my_tool(param: str) -> Dict[str, Any]:
        return {"result": f"Processed: {param}"}
    
    # Define schema
    schema = {
        "type": "object",
        "properties": {
            "param": {"type": "string", "minLength": 1}
        },
        "required": ["param"]
    }
    
    # Register tool
    mcp_instance.register_tool(
        name="my_tool",
        func=my_tool,
        schema=schema,
        description="My custom tool",
        module="my_module",
        category="custom",
        version="1.0.0",
        tags=["custom"],
        examples=[{"param": "example"}],
        timeout=30.0,
        rate_limit=10.0,
        cache_ttl=300.0
    )
    
    # Register resource
    def my_resource_retriever(uri: str) -> Dict[str, Any]:
        return {"content": f"Resource: {uri}", "uri": uri}
    
    mcp_instance.register_resource(
        uri_template="my://{resource_id}",
        retriever=my_resource_retriever,
        description="My custom resource",
        module="my_module",
        category="custom",
        version="1.0.0",
        cacheable=True
    )

# Module metadata
__version__ = "1.0.0"
__description__ = "My custom MCP module"
__dependencies__ = ["some_dependency"]
```

### Module Discovery

Modules are automatically discovered during initialization:

```python
# Discover modules
success = mcp_instance.discover_modules()

# Get module information
module_info = mcp_instance.get_module_info("my_module")
```

## Testing

### Running Tests

```bash
# Run all MCP tests
pytest src/tests/test_mcp_comprehensive.py -v

# Run specific test categories
pytest src/tests/test_mcp_comprehensive.py::TestMCPCoreComprehensive -v
pytest src/tests/test_mcp_comprehensive.py::TestMCPTransportLayers -v
```

### Test Categories

- **Unit Tests**: Core functionality testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Performance and load testing
- **Error Handling Tests**: Error scenarios and edge cases

## Troubleshooting

### Common Issues

#### Module Loading Failures
```python
# Check module status
status = mcp_instance.get_enhanced_server_status()
modules = status["modules"]

for module_name, module_info in modules.items():
    if module_info["status"] != "loaded":
        print(f"Module {module_name}: {module_info['status']}")
        if module_info.get("error_message"):
            print(f"  Error: {module_info['error_message']}")
```

#### Performance Issues
```python
# Check performance metrics
status = mcp_instance.get_enhanced_server_status()
performance = status["performance"]

if performance["success_rate"] < 0.9:
    print("Low success rate detected")
    
if performance["average_execution_time"] > 5.0:
    print("High execution times detected")
    
if performance["cache_hit_ratio"] < 0.5:
    print("Low cache efficiency")
```

#### Memory Issues
```python
# Check memory usage
status = mcp_instance.get_enhanced_server_status()
resources = status["resources"]

if resources["cache_size"] > 1000:
    print("Large cache detected, consider clearing")
    mcp_instance.clear_cache()
```

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp")
```

### Performance Optimization

#### Caching Strategy
```python
# Use caching for expensive operations
mcp_instance.register_tool(
    name="expensive_operation",
    func=expensive_function,
    schema=schema,
    cache_ttl=3600.0  # Cache for 1 hour
)
```

#### Rate Limiting
```python
# Prevent abuse with rate limiting
mcp_instance.register_tool(
    name="api_call",
    func=api_function,
    schema=schema,
    rate_limit=10.0  # 10 requests per second
)
```

#### Concurrent Limits
```python
# Limit resource usage
mcp_instance.register_tool(
    name="resource_intensive",
    func=intensive_function,
    schema=schema,
    max_concurrent=2  # Only 2 concurrent executions
)
```

## Security Considerations

### Authentication
```python
# Tools requiring authentication
mcp_instance.register_tool(
    name="sensitive_operation",
    func=sensitive_function,
    schema=schema,
    requires_auth=True
)
```

### Input Validation
```python
# Always enable input validation
mcp_instance.register_tool(
    name="safe_tool",
    func=safe_function,
    schema=schema,
    input_validation=True,
    output_validation=True
)
```

### Rate Limiting
```python
# Prevent abuse
mcp_instance.register_tool(
    name="public_api",
    func=public_function,
    schema=schema,
    rate_limit=5.0  # 5 requests per second
)
```

## Best Practices

### Tool Design
1. **Clear Naming**: Use descriptive, consistent tool names
2. **Comprehensive Schemas**: Define detailed parameter schemas
3. **Error Handling**: Provide meaningful error messages
4. **Documentation**: Include clear descriptions and examples
5. **Validation**: Enable input and output validation

### Performance
1. **Caching**: Use caching for expensive operations
2. **Rate Limiting**: Implement appropriate rate limits
3. **Concurrent Limits**: Set reasonable concurrent execution limits
4. **Monitoring**: Monitor performance metrics regularly
5. **Optimization**: Profile and optimize slow operations

### Security
1. **Authentication**: Require authentication for sensitive operations
2. **Validation**: Always validate inputs and outputs
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **Logging**: Log security-relevant events
5. **Updates**: Keep dependencies updated

### Module Development
1. **Structure**: Follow the standard module structure
2. **Metadata**: Provide complete module metadata
3. **Error Handling**: Handle errors gracefully
4. **Testing**: Include comprehensive tests
5. **Documentation**: Document all tools and resources

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints throughout
- Include comprehensive docstrings
- Write clear, readable code

### Testing
- Write unit tests for all new functionality
- Include integration tests for complex workflows
- Test error scenarios and edge cases
- Maintain good test coverage

## Conclusion

The Enhanced MCP implementation provides a comprehensive, production-ready solution for exposing GNN capabilities through the Model Context Protocol. With advanced features like caching, rate limiting, performance monitoring, and enhanced error handling, it offers a robust foundation for building AI-powered applications that can interact with GNN models and tools.

For more information, see the individual module documentation and the main project documentation. 