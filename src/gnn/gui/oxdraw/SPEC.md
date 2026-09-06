# oxdraw — Technical Specification

**Version**: 1.6.0

## Purpose

Visual diagram-as-code interface: bidirectional GNN ↔ Mermaid conversion with optional
interactive editing through the external `oxdraw` Rust CLI.

## Features

- GNN → Mermaid conversion with embedded metadata
- Mermaid → GNN parsing with visual-edit preservation
- Interactive editing via `oxdraw` CLI (recovery: headless conversion only)
- MCP tool integration

## Architecture

```
oxdraw/
├── __init__.py            # Public API (oxdraw_gui, process_oxdraw, converters)
├── processor.py           # process_oxdraw, check_oxdraw_installed, launch_oxdraw_editor
├── mermaid_converter.py   # GNN → Mermaid (gnn_to_mermaid, convert_gnn_file_to_mermaid)
├── mermaid_parser.py      # Mermaid → GNN (mermaid_to_gnn, convert_mermaid_file_to_gnn)
├── utils.py               # Node-shape / edge-style inference, syntax validation
└── mcp.py                 # MCP tool registration
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SKILL](SKILL.md)**: Capability API
