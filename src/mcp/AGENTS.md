# MCP Module - Agent Scaffolding

## Module Overview

**Purpose**: Model Context Protocol implementation for standardized tool discovery, registration, and execution across all GNN modules

**Pipeline Step**: Step 21: Model Context Protocol processing (21_mcp.py)

**Category**: Protocol Integration / Tool Management

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. Implement Model Context Protocol (MCP) for tool registration and discovery
2. Provide standardized interface for tool execution across modules
3. Enable inter-module communication and resource sharing
4. Manage MCP server lifecycle and client connections
5. Support multiple MCP transport protocols (stdio, HTTP, WebSocket)

### Key Capabilities
- Tool registration and discovery system
- Resource access and management
- JSON-RPC protocol implementation
- Server and client implementations
- Enhanced error handling and validation
- Performance monitoring and caching
- Concurrent execution control

---

## API Reference

### Public Functions

#### `process_mcp(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main MCP processing function called by orchestrator (21_mcp.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for MCP results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance for progress reporting (default: None)
- `mcp_mode` (str): MCP mode ("tool_discovery", "server", "client", default: "tool_discovery")
- `enable_tools` (bool): Enable MCP tools functionality (default: True)
- `**kwargs`: Additional MCP options

**Returns**: `True` if MCP processing succeeded

**Example**:
```python
from mcp import process_mcp

success = process_mcp(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/21_mcp_output"),
    verbose=True,
    mcp_mode="tool_discovery",
    enable_tools=True
)
```

#### `register_module_tools(module_name, tools) -> bool`
**Description**: Register tools from a specific module in the MCP system

**Parameters**:
- `module_name` (str): Name of the module registering tools
- `tools` (List[Dict]): List of tool definitions

**Returns**: `True` if registration succeeded

#### `get_available_tools() -> List[Dict]`
**Description**: Get list of all available MCP tools across all modules

**Returns**: List of tool information dictionaries

---

## MCP Protocol Implementation

### Core Classes

#### `MCP` - Main Protocol Class
**Description**: Core Model Context Protocol implementation

**Key Methods**:
- `initialize()` - Initialize MCP server
- `register_tools()` - Register available tools
- `handle_request()` - Process incoming requests
- `list_tools()` - List available tools

#### `MCPTool` - Tool Definition Class
**Description**: Represents a registered MCP tool

**Attributes**:
- `name` - Tool name
- `description` - Tool description
- `input_schema` - JSON schema for inputs
- `handler` - Function to execute tool

#### `MCPResource` - Resource Definition Class
**Description**: Represents accessible MCP resources

**Attributes**:
- `uri` - Resource URI
- `name` - Resource name
- `description` - Resource description
- `mime_type` - Resource MIME type

### Transport Protocols

#### STDIO Transport
- **Purpose**: Standard input/output communication
- **Use Case**: Local tool execution and testing
- **Implementation**: `mcp.server_stdio`

#### HTTP Transport
- **Purpose**: Web-based MCP server
- **Use Case**: Remote tool access and web integration
- **Implementation**: `mcp.server_http`

#### WebSocket Transport
- **Purpose**: Real-time bidirectional communication
- **Use Case**: Live tool execution and streaming results
- **Implementation**: `mcp.server_websocket`

---

## Dependencies

### Required Dependencies
- `json` - JSON-RPC protocol implementation
- `pathlib` - Path and URI handling
- `typing` - Type annotations and validation
- `logging` - Request/response logging

### Optional Dependencies
- `aiohttp` - HTTP server implementation (fallback: basic HTTP)
- `websockets` - WebSocket server (fallback: polling-based)
- `fastapi` - REST API framework (fallback: basic HTTP)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `MCP_SERVER_PORT` - MCP server port (default: 8080)
- `MCP_TRANSPORT` - Transport protocol ("stdio", "http", "websocket")
- `MCP_LOG_LEVEL` - MCP logging level ("DEBUG", "INFO", "WARNING", "ERROR")
- `MCP_TIMEOUT` - MCP request timeout (default: 30 seconds)

### Configuration Files
- `mcp_config.yaml` - MCP server and tool configuration

### Default Settings
```python
DEFAULT_MCP_SETTINGS = {
    'server': {
        'port': 8080,
        'host': 'localhost',
        'transport': 'stdio',
        'timeout': 30,
        'max_concurrent_requests': 10
    },
    'tools': {
        'auto_register': True,
        'validate_schemas': True,
        'cache_results': True,
        'rate_limiting': True
    },
    'logging': {
        'level': 'INFO',
        'format': 'json',
        'include_request_id': True
    }
}
```

---

## Usage Examples

### Basic MCP Processing
```python
from mcp.processor import process_mcp

success = process_mcp(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/21_mcp_output"),
    logger=logger,
    mcp_mode="tool_discovery"
)
```

### Tool Registration
```python
from mcp import register_module_tools

tools = [
    {
        'name': 'gnn_parse',
        'description': 'Parse GNN model files',
        'handler': parse_gnn_file,
        'input_schema': {...}
    }
]

success = register_module_tools('gnn', tools)
```

### Tool Discovery
```python
from mcp import get_available_tools

tools = get_available_tools()
for tool in tools:
    print(f"Tool: {tool['name']} - {tool['description']}")
```

---

## Output Specification

### Output Products
- `mcp_processing_summary.json` - MCP processing summary
- `registered_tools.json` - All registered tools information
- `mcp_server_status.json` - Server status and configuration
- `tool_execution_log.json` - Tool execution history

### Output Directory Structure
```
output/21_mcp_output/
├── mcp_processing_summary.json
├── registered_tools.json
├── mcp_server_status.json
└── tool_execution_log.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds (tool registration)
- **Memory**: ~10-20MB for tool registry
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: <1s for tool discovery
- **Slow Path**: ~5s for comprehensive tool validation
- **Memory**: ~5-15MB for typical tool sets

---

## Error Handling

### Graceful Degradation
- **No transport libraries**: Fallback to stdio-only mode
- **Tool registration failures**: Continue with available tools
- **Server startup failures**: Fallback to client-only mode

### Error Categories
1. **Protocol Errors**: Invalid JSON-RPC requests/responses
2. **Tool Errors**: Tool execution failures or timeouts
3. **Transport Errors**: Network or I/O communication failures
4. **Validation Errors**: Invalid tool schemas or parameters

---

## Integration Points

### Orchestrated By
- **Script**: `21_mcp.py` (Step 21)
- **Function**: `process_mcp()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_mcp_integration.py` - MCP integration tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Module Tools → MCP Registration → Tool Discovery → Execution Requests → Response Handling
```

---

## Testing

### Test Files
- `src/tests/test_mcp_integration.py` - Integration tests
- `src/tests/test_mcp_protocol.py` - Protocol tests
- `src/tests/test_mcp_tools.py` - Tool registration tests

### Test Coverage
- **Current**: 82%
- **Target**: 90%+

### Key Test Scenarios
1. Tool registration and discovery across modules
2. JSON-RPC protocol compliance
3. Multiple transport protocol operation
4. Error handling with malformed requests
5. Performance under concurrent tool execution

---

## MCP Integration

### Tools Registered (Across All Modules)
- `gnn_*` - GNN file processing tools
- `analysis_*` - Statistical analysis tools
- `visualization_*` - Visualization generation tools
- `render_*` - Code generation tools
- `gui_*` - GUI interaction tools

### Tool Categories
- **File Processing**: GNN parsing, validation, transformation
- **Analysis**: Statistical analysis, complexity metrics
- **Visualization**: Chart generation, interactive displays
- **Code Generation**: Multi-framework code rendering
- **Model Management**: Registry operations, version control

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready
