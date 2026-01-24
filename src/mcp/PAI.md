# MCP Module - PAI Context

## Quick Reference

**Purpose:** Model Context Protocol server for exposing pipeline tools to AI assistants.

**When to use this module:**
- Expose pipeline functions as MCP tools
- Enable AI-driven pipeline execution
- Provide structured tool interfaces

## Common Operations

```python
# Start MCP server
from mcp.server import MCPServer
server = MCPServer()
server.start()

# Register tools
server.register_tool("parse_gnn", parse_gnn_function)
server.register_tool("run_simulation", run_simulation_function)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | AI clients | Tool calls |
| **Output** | Pipeline | Executed operations |

## Key Files

- `server.py` - MCP server implementation
- `tools.py` - Tool definitions
- `processor.py` - MCP processing
- `__init__.py` - Public API exports

## Available Tools

| Tool | Description |
|------|-------------|
| `parse_gnn` | Parse GNN file |
| `render_code` | Generate framework code |
| `run_simulation` | Execute simulation |
| `analyze_results` | Analyze outputs |

## Tips for AI Assistants

1. **Step 21:** MCP processing is Step 21
2. **Protocol:** Follows Model Context Protocol spec
3. **Output Location:** `output/21_mcp_output/`
4. **Tools:** Exposes pipeline as callable functions
5. **AI Integration:** Enables Claude/GPT to run pipeline operations

---

**Version:** 1.1.3 | **Step:** 21 (MCP Integration)
