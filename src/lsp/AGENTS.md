# LSP Module — Agent Scaffolding

## Module Overview

**Purpose**: Language Server Protocol implementation for real-time diagnostics and hover information for GNN files.
**Pipeline Step**: Infrastructure module (not a numbered step)
**Category**: Infrastructure / Development Tools
**Status**: ✅ Production Ready
**Version**: 1.6.0
**Last Updated**: 2026-04-16

The LSP module implements a GNN Language Server using the Language Server Protocol. It provides real-time diagnostics and hover information for GNN model files in any LSP-compatible editor.

## Architecture

- **Pattern**: Infrastructure module (not a numbered pipeline step)
- **Framework**: `pygls` (Python Language Server)
- **Protocol**: LSP 3.x over stdio
- **Dependency**: Optional (`pygls` — graceful fallback when missing)

## Capabilities

- **Diagnostic publishing**: Validates GNN sections, state-space variables, and connections on file open/save
- **Hover information**: Displays variable name, dimensions, type, and default values at cursor position
- **Error extraction**: Maps parse errors to line numbers for inline display

## Key Functions

| Function | Description |
|----------|-------------|
| `create_server()` | Configure and return a `LanguageServer` instance |
| `start_server()` | Start the server on stdio |
| `_publish_diagnostics()` | Run GNN validation and publish results |
| `_get_hover()` | Generate hover info for a variable at a position |

## File Structure

```
lsp/
├── __init__.py    # Full server implementation (226 lines)
├── AGENTS.md      # This file
├── README.md      # Usage guide
└── SPEC.md        # Module specification
```

## References

- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Specification

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
