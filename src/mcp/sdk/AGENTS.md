# Sdk - Agent Scaffolding

## Module Overview

**Purpose**: Responsible for `Sdk` operations within the GNN pipeline architecture.
**Category**: Generated Pipeline Component
**Status**: Development

---

## Core Functionality

### Primary Responsibilities
MCP SDK server shim: delegates to the parent mcp.server implementation.
Present so MCPSDKStatus health check finds a complete SDK under src/mcp/sdk/. MCP SDK client shim: minimal client interface delegating to the parent mcp module.
Present so MCPSDKStatus health check finds a complete SDK under src

### Extracted Code Entities

- **Classes**: No specific classes exported.
- **Functions**: No specific public functions exported.

## Implementation Details

This module follows the Thin Orchestrator Pattern. It is governed by the Zero-Mock testing policy.
