# GNN Model Context Protocol Implementation Specification

## Purpose

This document defines the maintained MCP implementation used by the GNN
pipeline. The executable registry is `src/mcp/mcp.py`; JSON-RPC request
handling lives in `src/mcp/server_core.py` and is re-exported through
`src/mcp/server.py`.

## Runtime Surfaces

- `MCP`: core registry for tools, resources, modules, and performance metrics.
- `MCPRegistry`: public alias for the core registry type.
- `MCPServer` / `JSONRPCServer`: JSON-RPC 2.0 request handler bound to an
  `MCP` instance.
- `initialize(...)`: configures the registry and discovers module tools.
- `register_tools(mcp)`: discovers per-module `mcp.py` files and returns
  whether registration completed without module loading failures.
- `create_mcp_server(mcp_instance=None)`: constructs a JSON-RPC server.

## Tool Registration Contract

Every module-level `register_tools(mcp_instance)` implementation must register
real callable functions with deterministic schemas:

```python
mcp_instance.register_tool(
    name="module.tool_name",
    func=callable_function,
    schema={"type": "object", "properties": {}},
    description="Action-oriented description.",
)
```

Registered functions must perform real work or explicit introspection. They must
not silently succeed when no work has occurred.

## Discovery Flow

1. `initialize(...)` obtains the shared registry.
2. The registry applies performance and validation options.
3. `MCP.discover_modules(...)` locates `src/<module>/mcp.py` files.
4. Each module's `register_tools` function is executed with timeout controls.
5. The registry records `MCPModuleInfo` for success and error states.
6. Initialization returns `(mcp_instance, sdk_found, all_modules_loaded)`.

## JSON-RPC Methods

`MCPServer` supports:

- `initialize`
- `notifications/initialized`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`
- `shutdown`
- `exit`

The server delegates tool execution to the bound `MCP` registry and returns
structured JSON-RPC responses.

## Verification

Focused checks:

```bash
uv run pytest src/tests/mcp/test_mcp_configurability.py src/tests/mcp/test_mcp_audit.py -q
uv run python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```
