# GNN MCP Client Setup Guide

Connect any MCP-compatible client to the GNN pipeline and access all 131 tools interactively.

**Last Updated**: March 6, 2026

## Prerequisites

```bash
# Ensure the GNN package is installed (from repo root)
pip install -e .
# or using uv:
uv pip install -e .

# Verify MCP server starts
PYTHONPATH=src python src/21_mcp.py --help
```

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gnn": {
      "command": "python",
      "args": ["/Users/4d/Documents/GitHub/generalizednotationnotation/src/21_mcp.py"],
      "env": {
        "PYTHONPATH": "/Users/4d/Documents/GitHub/generalizednotationnotation/src"
      }
    }
  }
}
```

Restart Claude Desktop. You will see **GNN** in the tools palette (🔧 icon). You can then ask Claude:

> *"Parse the file at input/gnn_files/actinf_pomdp_agent.md using parse_gnn_content"*
> *"Run process_validation on my GNN directory"*
> *"List all 131 GNN tools"*

## VS Code (via MCP extension)

1. Install the **MCP Servers** extension (marketplace ID: `anthropic.mcp`)
2. Open Settings → MCP → Add Server:

```json
{
  "name": "gnn",
  "transport": "stdio",
  "command": "python",
  "args": ["${workspaceFolder}/src/21_mcp.py"],
  "env": { "PYTHONPATH": "${workspaceFolder}/src" }
}
```

1. Reload VS Code. The GNN tools become available in Copilot Chat and inline suggestions.

## Cursor

Add to `.cursor/mcp.json` (repo root or home directory):

```json
{
  "mcpServers": {
    "gnn": {
      "command": "python",
      "args": ["src/21_mcp.py"],
      "env": { "PYTHONPATH": "src" }
    }
  }
}
```

Restart Cursor. In the AI panel, `@gnn` references tools from the GNN server.

## Using uv (recommended for isolated environments)

```json
{
  "mcpServers": {
    "gnn": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/Users/4d/Documents/GitHub/generalizednotationnotation",
        "python", "src/21_mcp.py"
      ]
    }
  }
}
```

## Verifying the Connection

After connecting, test with a discovery call:

```python
# In any Python MCP client:
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["src/21_mcp.py"],
    env={"PYTHONPATH": "src"}
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(f"Connected: {len(tools.tools)} tools available")
        # → Connected: 131 tools available
```

## Available Tools by Use Case

| I want to… | Tool to call |
|------------|--------------|
| Parse a GNN file | `parse_gnn_content` |
| Validate a GNN file | `validate_gnn_file` |
| Run the full pipeline | `get_pipeline_steps` then each step's `process_*` |
| Execute with PyMDP | `execute_pymdp_simulation` |
| Ask LLM to analyse | `analyze_gnn_with_llm` |
| Generate audio from GNN | `process_audio` |
| Export to multiple formats | `process_export` |
| Check security | `scan_gnn_file` |

## Troubleshooting

**Server won't start**: Ensure `PYTHONPATH=src` includes the `src/` directory where all modules live.

**Tools not appearing**: Run the audit to confirm all 131 tools register cleanly:

```bash
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py -v --tb=short
```

**Timeout on heavy steps**: Steps 12 (execute) and 13 (LLM) can take minutes. Set client timeout to ≥300s.

## See Also

- [Tool Reference](tool_reference.md) — all 131 tools in a flat table
- [Tool Development Guide](tool_development_guide.md) — add your own tools
- [modules/21_mcp.md](../modules/21_mcp.md) — pipeline step documentation
- [doc/mcp/fastmcp.md](../../../doc/mcp/fastmcp.md) — FastMCP library guide
