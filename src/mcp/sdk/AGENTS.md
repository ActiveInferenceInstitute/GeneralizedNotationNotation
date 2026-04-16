# MCP SDK Shim

## Overview

Provides a thin SDK compatibility layer that delegates to the parent `mcp` module implementation. Present so that `MCPSDKStatus` health checks find a complete SDK surface under `src/mcp/sdk/`.

## Architecture

```
sdk/
├── client.py     # Client-side MCP SDK shim (16 lines)
├── mcp.py        # Core SDK re-exports from parent mcp module (39 lines)
└── server.py     # Server-side MCP SDK shim (16 lines)
```

## Purpose

- **Health check compliance** — `MCPSDKStatus` validates SDK presence by checking this directory.
- **API surface delegation** — All exports delegate to `mcp.mcp.MCP` and related classes.
- **No independent logic** — This is a pure re-export layer, not an independent implementation.

## Parent Module

See [mcp/AGENTS.md](../AGENTS.md) for the full MCP architecture.

**Version**: 1.6.0
