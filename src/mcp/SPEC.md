# MCP Module Specification

## Overview
Model Context Protocol server and client implementation for GNN integration.

## Components

### Core
- `mcp.py` - Main MCP server (2004 lines)
- `exceptions.py` - 11 exception classes (182 lines)

### Clients
- `sympy_mcp_client.py` - SymPy integration with stochasticity/stability analysis
- `npx_inspector.py` - NPX package inspection

## Exception Classes
- `MCPError`, `MCPToolNotFoundError`, `MCPResourceNotFoundError`
- `MCPParameterValidationError`, `MCPExecutionError`

## Key Exports
```python
from mcp import MCPServer, process_mcp
from mcp.exceptions import MCPError, MCPToolNotFoundError
```

## Testing
Tests in `tests/test_mcp_overall.py` (10 tests)
