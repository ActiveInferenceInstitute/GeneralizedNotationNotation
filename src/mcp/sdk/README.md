# MCP SDK Shim

Thin compatibility layer delegating to the parent `mcp` module. Exists for `MCPSDKStatus` health check compliance.

## Files

- `client.py` — Client-side shim (re-exports)
- `mcp.py` — Core API re-exports from `mcp.mcp`
- `server.py` — Server-side shim (re-exports)

## Note

This is a **pure delegation layer** — no independent logic. All real implementation lives in `src/mcp/mcp.py`.

## See Also

- [Parent: mcp/README.md](../README.md)
