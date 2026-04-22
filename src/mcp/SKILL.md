---
name: gnn-mcp-protocol
description: "GNN Model Context Protocol server and tool registration. Use when registering GNN pipeline operations as MCP tools, starting the MCP JSON-RPC server, configuring tool discovery, or exposing GNN capabilities to LLM agents via tool-use."
---

# GNN MCP Protocol (Step 21)

## Purpose

Processes Model Context Protocol (MCP) configurations and registers GNN pipeline operations as MCP tools. Enables LLM agents to invoke GNN capabilities through standardized tool interfaces.

## Key Commands

```bash
# Run MCP processing
python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 21 --verbose
```

## API

```python
from mcp import (
    MCP, MCPServer, JSONRPCServer, MCPTool, MCPResource,
    create_mcp_server, start_mcp_server,
    register_tools, register_module_tools,
    get_available_tools, list_available_tools,
    list_available_resources, process_mcp,
    get_mcp_instance, handle_mcp_request,
    initialize,
)

# Process MCP step (used by pipeline)
process_mcp(
    target_dir, output_dir,
    verbose=True,
    performance_mode="high",
    strict_validation=True,
    cache_ttl=120.0,
)

# Create and start a JSON-RPC server bound to the global MCP registry
server = create_mcp_server()      # returns JSONRPCServer
start_mcp_server()

# Discover and register all pipeline modules' tools on the global MCP
register_tools()                             # populates mcp_instance
register_module_tools("gnn")                 # named module only

# Query available tools and resources
tools     = get_available_tools()            # same as list_available_tools()
tools_md  = list_available_tools()
resources = list_available_resources()
```

## Key Exports

- `MCP` — core registry (tools, resources, modules, performance metrics).
- `MCPServer` — alias of `MCP` (legacy; the registry itself, not a network server).
- `JSONRPCServer` — JSON-RPC 2.0 request handler bound to an `MCP` instance.
- `MCPTool` / `MCPResource` — dataclasses backing the registry.
- `create_mcp_server` / `start_mcp_server` — construct and start a JSON-RPC server.
- `register_tools` — discover and register all pipeline modules' tools on the global MCP.
- `register_module_tools` — register a single named module.
- `initialize` — check SDK + discover modules; exposes `performance_mode`,
  `enable_caching`, `enable_rate_limiting`, `strict_validation`, `cache_ttl`,
  `modules_allowlist`, `per_module_timeout`, `overall_timeout`, `force_refresh`.
- `process_mcp` — pipeline entry point (forwards overrides to `initialize`).
- `get_available_tools`, `list_available_tools`, `list_available_resources`.
- Error types: `MCPToolNotFoundError`, `MCPValidationError`, `MCPModuleLoadError`,
  `MCPInvalidParamsError`, `MCPToolExecutionError`, `MCPSDKNotFoundError`,
  `MCPResourceNotFoundError`, `MCPPerformanceError`.

## Configuration

The `MCP` singleton honours these knobs, all propagated through
`initialize()` and `process_mcp(**kwargs)`:

| Knob | Type | Default | Purpose |
|------|------|---------|---------|
| `performance_mode` | `"low"`/`"high"` | `"low"` | Bulk toggle for caching, rate limiting, strict validation |
| `enable_caching` | `bool` | from mode | Enable result cache |
| `enable_rate_limiting` | `bool` | from mode | Enable per-tool rate limiting |
| `strict_validation` | `bool` | from mode | Enforce JSON-schema validation of params |
| `cache_ttl` | `float` (sec) | `300.0` | Result-cache TTL |
| `modules_allowlist` | `list[str]` | `None` | Only load these modules under `src/` |
| `per_module_timeout` | `float` | `30.0` | Max seconds per module during discovery |
| `overall_timeout` | `float` | `120.0` | Wall-clock budget for parallel discovery |
| `force_refresh` | `bool` | `False` | Re-discover modules even if already loaded |

## Recommended Workflow

```python
from mcp import initialize, register_tools, get_available_tools, create_mcp_server

# 1. Initialize the MCP registry
initialize(performance_mode="high", strict_validation=True)

# 2. Discover and register all pipeline module tools
register_tools()

# 3. Verify tools registered correctly
tools = get_available_tools()
assert len(tools) > 0, "No tools registered — check module discovery"

# 4. Start the JSON-RPC server
server = create_mcp_server()
```

## Output

- MCP configuration in `output/21_mcp_output/`
- Tool registration manifests
- Server configuration files

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [MCP Protocol Spec](https://modelcontextprotocol.io/)
