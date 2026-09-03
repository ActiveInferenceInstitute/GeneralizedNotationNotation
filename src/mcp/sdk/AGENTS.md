# MCP SDK Facade

## Overview

Provides a thin SDK facade that delegates to the parent `mcp` module implementation. Present so that `MCPSDKStatus` health checks find a complete SDK surface under `src/mcp/sdk/`.

## Architecture

```
sdk/
├── client.py     # Client-side MCP SDK facade (re-exports)
├── mcp.py        # Core SDK re-exports from parent mcp module
└── server.py     # Server-side MCP SDK facade (re-exports)
```

## Purpose

- **Health check compliance** — `MCPSDKStatus` validates SDK presence by checking this directory.
- **API surface delegation** — All exports delegate to `mcp.mcp.MCP` and related classes.
- **No independent logic** — This is a pure re-export layer, not an independent implementation.

## Parent Module

See [mcp/AGENTS.md](../AGENTS.md) for the full MCP architecture.

**Version**: 3.2.0
