# GNN Model Context Protocol Hub

**Last Updated**: February 24, 2026  
**Status**: ✅ Production Ready (76 real tools, 203 audit tests passing)

The GNN pipeline exposes every module as a fully-tested MCP tool suite, enabling any MCP-compatible AI client (Claude Desktop, VS Code, Cursor, etc.) to drive the entire pipeline interactively.

## Quick Links

| Document | Description |
|----------|-------------|
| **[modules/21_mcp.md](../modules/21_mcp.md)** | Pipeline step: architecture, 76-tool registry, audit commands |
| **[tool_reference.md](tool_reference.md)** | Flat quick-reference table — all 76 tools, domain, description |
| **[client_setup.md](client_setup.md)** | Connect Claude Desktop, VS Code, Cursor to the GNN MCP server |
| **[tool_development_guide.md](tool_development_guide.md)** | Add a new MCP tool to any pipeline module |
| **[../../mcp/gnn_mcp_model_context_protocol.md](../../mcp/gnn_mcp_model_context_protocol.md)** | Deep-dive: MCP protocol spec, JSON-RPC 2.0, GNN-MCP integration theory |
| **[../../mcp/fastmcp.md](../../mcp/fastmcp.md)** | FastMCP library guide used by the GNN server |

## Architecture at a Glance

```
AI Client (Claude Desktop / VS Code / Cursor)
        │  JSON-RPC 2.0 over stdio/HTTP
        ▼
  src/21_mcp.py  ─── discovers all module mcp.py files
        │
  ┌─────┴──────────────────────────────────────────┐
  │  per-module  mcp.py  (register_tools)           │
  │  22 domains × avg 3.5 tools = 76 total tools   │
  └─────────────────────────────────────────────────┘
```

## The 22 Tool Domains

`analysis` · `audio` · `advanced_visualization` · `execute` · `export` · `gnn` · `gui` · `integration` · `intelligent_analysis` · `llm` · `ml_integration` · `ontology` · `pipeline` · `render` · `report` · `research` · `sapf` · `security` · `utils` · `validation` · `visualization` · `website`

## Audit Quick-Check

```bash
# Run 203-test MCP audit (0 stubs, 0 skips)
PYTHONPATH=src python -m pytest src/tests/test_mcp_audit.py -v

# Generate JSON report of all 76 tools
PYTHONPATH=src python src/mcp/validate_tools.py
```

## See Also

- [GNN Overview](../gnn_overview.md) — what GNN is
- [Pipeline Modules](../modules/README.md) — all 25 pipeline steps
- [Technical Reference](../technical_reference.md) — architecture details
