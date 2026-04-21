# Specification: Model Context Protocol (MCP) Documentation

## Scope
Documentation for the GNN MCP server: tool catalog, wiring contracts, and
development guide for adding new tools. Complements `src/mcp/` which is
the executable server implementation.

## Contents
| File | Purpose |
|------|---------|
| `README.md` | MCP server overview + quick start |
| `tool_reference.md` | Canonical registry of all 131 real tools |
| `tool_development_guide.md` | How to add a new @mcp_tool |

## Tool Contract
Every MCP tool implements:
1. A standard signature: `(arg1, arg2, ...) -> Dict[str, Any]`
2. A `success: bool` key in the returned dict
3. Schema declaration via `register_tool(name, func, schema, description)`
4. Zero-mock: no MagicMock in tests — real deps or skip-with-guard

## Versioning
Tool reference is tagged with the engine version (currently v1.6.0).
Tools added post-release MUST update `tool_reference.md` before merging.

## Status
Maintained. When adding a tool, follow
[`tool_development_guide.md`](tool_development_guide.md) and update
[`tool_reference.md`](tool_reference.md).
