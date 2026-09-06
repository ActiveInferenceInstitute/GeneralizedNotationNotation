---
name: gnn-oxdraw
description: oxdraw visual editor integration for GNN. Use when launching the oxdraw interactive visual editor, converting GNN files to Mermaid flowchart format, or compiling edited Mermaid files back into GNN markup.
---

# oxdraw Visual Editor Integration

## Purpose

The `oxdraw` module coordinates bidirectional synchronization between text-based GNN markdown definitions and interactive visual flowcharts (Mermaid format), allowing users to visually construct and edit Active Inference models using the `cargo` installed `oxdraw` Rust editor.

## Key APIs

- Mermaid-to-GNN conversion
- GNN-to-Mermaid conversion
- oxdraw editor proxy

## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `oxdraw.check_installation`
- `oxdraw.convert_from_mermaid`
- `oxdraw.convert_to_mermaid`
- `oxdraw.get_info`
- `oxdraw.launch_editor`

## References

- [AGENTS.md](../AGENTS.md) — Parent GUI overview
- [../../README.md](../../README.md) — Root documentation
