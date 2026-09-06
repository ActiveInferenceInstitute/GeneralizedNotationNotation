---
name: gnn-gui-1
description: Core Graphical User Interface module for GNN. Use when querying GUI availability, checking GUI export paths, and managing the main interactive interface lifecycle.
---

# GNN GUI 1 (Core GUI Submodule)

## Purpose

The `gui_1` module provides the form-based Interactive GNN Constructor: a Gradio
web interface for building and editing GNN models with live markdown synchronization
and headless artifact generation when Gradio is unavailable.

## Key APIs

- `gui_1(target_dir, output_dir, logger, **kwargs)` — main entry point
- `get_gui_1_info()` — capability metadata for the aggregator

## MCP Tools

`mcp.py` exposes a discovery hook (`register_gui_tools`) that triggers tool
registration on the parent `gui` module; the `gui_1` submodule itself defines
no standalone tools.

## References

- [AGENTS.md](../AGENTS.md) — Parent GUI overview
- [../../README.md](../../README.md) — Root documentation
