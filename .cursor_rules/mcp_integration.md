# MCP (Model Context Protocol) Integration

### Universal MCP Pattern
Every applicable module includes `mcp.py` with:
- **Tool Registration**: Module-specific MCP tool definitions
- **Function Exposure**: Key module functions exposed as MCP tools
- **Standardized Interface**: Consistent MCP API across all modules
- **Real Integration**: Functional MCP tools, not placeholder implementations

### MCP System Architecture
- **Central Registry**: `src/mcp/` contains core MCP implementation
- **Module Integration**: Each module's `mcp.py` registers with central system
- **Tool Discovery**: Automatic tool discovery and registration
- **API Consistency**: Standardized request/response patterns 