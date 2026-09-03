# MCP Module Specification

## Overview
Model Context Protocol server and client implementation for GNN integration.

## Components

### Core
- `mcp.py` - Main MCP server (`MCP`, `MCPTool`, `MCPResource`, discovery, execution)
- `exceptions.py` - MCP exception classes (see below)

### Clients
- `sympy_mcp_client.py` - SymPy integration with stochasticity/stability analysis
- `npx_inspector.py` - NPX package inspection

## Exception Classes
`MCPError` (base), `MCPToolNotFoundError`, `MCPResourceNotFoundError`,
`MCPInvalidParamsError`, `MCPToolExecutionError`, `MCPValidationError`,
`MCPSDKNotFoundError`, `MCPModuleLoadError`, `MCPPerformanceError`,
`MCPRateLimitError`, `MCPCacheError`, `MCPModuleDiscoveryError`.

## Key Exports
```python
from mcp import MCPServer, process_mcp
from mcp.exceptions import MCPError, MCPToolNotFoundError
```

## Testing
Tests in `src/tests/mcp/` (`test_mcp_overall.py` and siblings).



---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
