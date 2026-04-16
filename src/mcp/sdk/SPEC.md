# MCP SDK Shim — Technical Specification

**Version**: 1.6.0

## Purpose

Compatibility shim for `MCPSDKStatus` health check. No independent logic.

## Delegation Pattern

All exports forward to `mcp.mcp.MCP` and related parent module classes.

## Health Check Contract

`MCPSDKStatus` checks:
1. `sdk/` directory exists
2. `sdk/client.py`, `sdk/server.py`, `sdk/mcp.py` are importable
3. Core exports (`MCP`, `get_mcp_instance`) are accessible
