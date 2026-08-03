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

`mcp.py` exposes a discovery hook (`register_gui_tools`) that triggers tool
registration on the parent `gui` module; the `gui_1` submodule itself defines
no standalone tools.

## References

- [AGENTS.md](../AGENTS.md) — Parent GUI overview
- [../../README.md](../../README.md) — Root documentation
