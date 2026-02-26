---
name: gnn-mcp-protocol
description: GNN Model Context Protocol processing and tool registration. Use when registering GNN operations as MCP tools, building MCP server configurations, or integrating GNN capabilities with LLM tool-use workflows.
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
    MCP, MCPServer, MCPTool, MCPResource,
    create_mcp_server, start_mcp_server,
    register_tools, register_module_tools,
    get_available_tools, list_available_tools,
    list_available_resources, process_mcp,
    get_mcp_instance, handle_mcp_request
)

# Process MCP step (used by pipeline)
process_mcp(target_dir, output_dir, verbose=True)

# Create and start MCP server
server = create_mcp_server()
start_mcp_server(server)

# Register tools
register_tools(server)
register_module_tools(server, module_name="gnn")

# Query available tools
tools = get_available_tools()
tools_list = list_available_tools()
resources = list_available_resources()
```

## Key Exports

- `MCP` / `MCPServer` — server classes
- `MCPTool` / `MCPResource` — tool and resource types
- `create_mcp_server` / `start_mcp_server` — server lifecycle
- `register_tools` / `register_module_tools` — tool registration
- `process_mcp` — main pipeline processing function
- `get_available_tools`, `list_available_tools`, `list_available_resources`
- Error types: `MCPToolNotFoundError`, `MCPValidationError`, `MCPModuleLoadError`

## Output

- MCP configuration in `output/21_mcp_output/`
- Tool registration manifests
- Server configuration files

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [MCP Protocol Spec](https://modelcontextprotocol.io/)


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
