---
name: gnn-gui-1
description: Core Graphical User Interface module for GNN. Use when querying GUI availability, checking GUI export paths, and managing the main interactive interface lifecycle.
---

# GNN GUI 1 (Core GUI Submodule)

## Purpose

The `gui_1` module provides the core web-based or local graphical user interface components for interacting with the GNN pipeline, rendering dashboards, and viewing pipeline execution statuses.

## Key APIs

- GUI Initialization
- Status monitoring
- Pipeline dashboard binding

## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `gui_status`

## References

- [AGENTS.md](../AGENTS.md) — Parent GUI overview
- [../../README.md](../../README.md) — Root documentation
