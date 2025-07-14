# Model Context Protocol (MCP) Implementation

This directory contains the comprehensive Model Context Protocol (MCP) implementation for the GeneralizedNotationNotation (GNN) project. The MCP server exposes all GNN functionalities as standardized tools that can be accessed by MCP-compatible clients.

## Overview

The GNN MCP implementation provides:
- **Core MCP Server**: JSON-RPC 2.0 compliant server with tool and resource management
- **Multiple Transport Layers**: stdio and HTTP transport support
- **Comprehensive Tool Ecosystem**: Tools from all GNN modules (gnn, type_checker, export, visualization, etc.)
- **Meta-Tools**: Server introspection and diagnostic capabilities
- **CLI Interface**: Command-line access to all MCP functionality
- **Extensible Architecture**: Easy addition of new tools and resources

## Architecture

```
src/mcp/
├── mcp.py                 # Core MCP server implementation
├── server_stdio.py        # stdio transport server
├── server_http.py         # HTTP transport server
├── cli.py                 # Command-line interface
├── meta_mcp.py           # Meta-tools for server introspection
├── sympy_mcp.py          # SymPy integration tools
├── sympy_mcp_client.py   # SymPy MCP client
├── npx_inspector.py      # NPX inspector utilities
├── README.md             # This documentation
└── *.md                  # Additional documentation files
```

## Core Components

### 1. MCP Server (`mcp.py`)

The central MCP server implementation that:
- Manages tool and resource registration
- Handles JSON-RPC 2.0 requests
- Provides module discovery and loading
- Implements error handling and logging
- Tracks performance metrics

**Key Features:**
- Dynamic module loading
- Tool and resource registration
- Performance tracking
- Error handling with custom MCP error codes
- Server status monitoring

### 2. Transport Servers

#### stdio Server (`server_stdio.py`)
- Reads JSON-RPC requests from stdin
- Writes responses to stdout
- Multi-threaded architecture for concurrent processing
- Ideal for local process communication

#### HTTP Server (`server_http.py`)
- HTTP-based JSON-RPC server
- Supports both GET and POST requests
- Configurable host and port
- Suitable for network-based access

### 3. Command Line Interface (`cli.py`)

Comprehensive CLI for MCP operations:
```bash
# List all capabilities
python -m src.mcp.cli list

# Execute a tool
python -m src.mcp.cli execute get_gnn_files --params '{"target_dir": "doc"}'

# Get server status
python -m src.mcp.cli status

# Start server
python -m src.mcp.cli server --transport stdio
python -m src.mcp.cli server --transport http --host 0.0.0.0 --port 8080
```

### 4. Meta-Tools (`meta_mcp.py`)

Server introspection and diagnostic tools:
- `get_mcp_server_capabilities`: Full server capabilities
- `get_mcp_server_status`: Operational status and metrics
- `get_mcp_server_auth_status`: Authentication configuration
- `get_mcp_server_encryption_status`: Encryption status
- `get_mcp_module_info`: Detailed module information
- `get_mcp_tool_categories`: Tools organized by category
- `get_mcp_performance_metrics`: Performance statistics

## Available Tools by Module

### GNN Module (`src/gnn/mcp.py`)
- GNN file discovery and parsing
- Model structure analysis
- Parameter extraction and validation

### Type Checker (`src/type_checker/mcp.py`)
- GNN syntax validation
- Resource estimation
- Type consistency checking

### Export (`src/export/mcp.py`)
- Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
- Network graph export
- Structured data preservation

### Visualization (`src/visualization/mcp.py`)
- Graph visualization
- Matrix visualization
- Ontology relationship diagrams

### Render (`src/render/mcp.py`)
- PyMDP code generation
- RxInfer.jl model translation
- Template-based code generation

### Execute (`src/execute/mcp.py`)
- Script execution
- Result capture and reporting
- Multi-backend support

### LLM (`src/llm/mcp.py`)
- AI-powered model analysis
- Enhancement suggestions
- Natural language explanations

### Site (`src/site/mcp.py`)
- HTML site generation
- Report aggregation
- Interactive elements

### SAPF (`src/sapf/mcp.py`)
- Audio generation and sonification
- Model sonification
- Real-time audio processing

### Pipeline (`src/pipeline/mcp.py`)
- Pipeline step discovery
- Execution monitoring
- Configuration management

### Utils (`src/utils/mcp.py`)
- System diagnostics
- File operations
- Environment validation

## Usage Examples

### 1. Starting the MCP Server

#### stdio Transport (Recommended for local use)
```bash
python -m src.mcp.cli server --transport stdio
```

#### HTTP Transport (For network access)
```bash
python -m src.mcp.cli server --transport http --host 0.0.0.0 --port 8080
```

### 2. Using the CLI

#### List all available tools
```bash
python -m src.mcp.cli list --format human
```

#### Execute a GNN tool
```bash
python -m src.mcp.cli execute get_gnn_files --params '{"target_dir": "doc", "recursive": true}'
```

#### Get server status
```bash
python -m src.mcp.cli status --format json
```

### 3. JSON-RPC API Usage

#### Get server capabilities
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "mcp.capabilities",
  "params": {}
}
```

#### Execute a tool
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "mcp.tool.execute",
  "params": {
    "name": "get_gnn_files",
    "params": {
      "target_dir": "doc",
      "recursive": true
    }
  }
}
```

#### Direct tool invocation
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "get_gnn_files",
  "params": {
    "target_dir": "doc",
    "recursive": true
  }
}
```

## Error Handling

The MCP implementation provides comprehensive error handling:

### Standard JSON-RPC Error Codes
- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

### Custom MCP Error Codes
- `-32000`: MCP-specific errors
- `-32001`: Tool execution errors
- `-32002`: Resource retrieval errors
- `-32003`: Module loading errors

### Error Response Format
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32001,
    "message": "Tool execution failed",
    "data": {
      "tool": "get_gnn_files",
      "details": "Target directory not found"
    }
  }
}
```

## Performance Monitoring

The MCP server tracks various performance metrics:
- Request count and error rates
- Average execution times per tool
- Module loading statistics
- Server uptime and activity

Access performance data via:
```bash
python -m src.mcp.cli execute get_mcp_performance_metrics
```

## Security Considerations

### Transport Security
- **stdio**: Local process only, high security
- **HTTP**: Network accessible, consider HTTPS for production

### Authentication
- No built-in authentication (relies on transport security)
- Implement authentication for network deployments
- Use stdio transport for maximum security

### Recommendations
1. Use stdio transport for local-only access
2. Configure HTTPS for HTTP transport if needed
3. Implement authentication for untrusted networks
4. Monitor access logs and performance metrics

## Development and Extension

### Adding New Tools

1. Create or update the module's `mcp.py` file
2. Implement tool functions with proper error handling
3. Register tools using `mcp_instance.register_tool()`
4. Add comprehensive documentation and schemas

### Example Tool Registration
```python
def register_tools(mcp_instance):
    mcp_instance.register_tool(
        name="my_tool",
        func=my_tool_function,
        schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": ["param1"]
        },
        description="Description of my tool",
        module="my_module",
        category="My Category",
        version="1.0.0"
    )
```

### Testing MCP Tools

Use the CLI to test tools:
```bash
# Test tool execution
python -m src.mcp.cli execute my_tool --params '{"param1": "value"}'

# Test tool info
python -m src.mcp.cli info my_tool
```

## Integration with External Clients

### Claude Desktop
Configure Claude Desktop to use the GNN MCP server:
```json
{
  "mcpServers": {
    "gnn": {
      "command": "python",
      "args": ["-m", "src.mcp.cli", "server", "--transport", "stdio"]
    }
  }
}
```

### Other MCP Clients
The server is compatible with any JSON-RPC 2.0 MCP client. Use the HTTP transport for network-based clients or stdio for local integration.

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

2. **Tool Execution Failures**
   - Check tool parameters and schemas
   - Review error messages and logs
   - Validate input data

3. **Server Connection Issues**
   - Verify transport configuration
   - Check firewall settings for HTTP transport
   - Ensure proper permissions

### Debug Mode

Enable verbose logging:
```bash
python -m src.mcp.cli --verbose list
```

### Log Files

Check log files in the output directory:
```bash
ls -la output/logs/
```

## Contributing

When contributing to the MCP implementation:

1. Follow the established patterns for tool registration
2. Add comprehensive error handling
3. Include proper documentation and schemas
4. Test with both stdio and HTTP transports
5. Update this README for new features

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [GNN Project Documentation](../doc/)
- [MCP Integration Guide](../doc/mcp/gnn_mcp_model_context_protocol.md) 