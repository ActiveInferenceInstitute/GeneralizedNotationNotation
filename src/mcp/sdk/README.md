# MCP SDK Facade

Thin delegation layer forwarding to the parent `mcp` module. Exists for `MCPSDKStatus` health check compliance.

## Files

- `client.py` — Client-side facade (re-exports)
- `mcp.py` — Core API re-exports from `mcp.mcp`
- `server.py` — Server-side facade (re-exports)

## Note

This is a **pure delegation layer** — no independent logic. All real implementation lives in `src/mcp/mcp.py`.

## See Also

- [Parent: mcp/README.md](../README.md)
