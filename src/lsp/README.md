# GNN Language Server (LSP)

## Overview

Minimal Language Server Protocol implementation for GNN files. Provides real-time diagnostics and hover information in editors that support LSP (VS Code, Neovim, etc.).

**Requires**: `pygls` package (`pip install pygls`)

## Features

| Feature | LSP Method | Description |
|---------|-----------|-------------|
| **Diagnostics on open** | `textDocument/didOpen` | Validates sections, state-space, and connections |
| **Diagnostics on save** | `textDocument/didSave` | Re-validates after changes |
| **Hover info** | `textDocument/hover` | Shows variable dimensions and type on hover |

## Usage

```bash
# Start via CLI
gnn lsp

# Or directly
python -m src.lsp
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "gnn-lsp.server.path": "gnn",
  "gnn-lsp.server.args": ["lsp"]
}
```

## Architecture

- **Server**: Built on `pygls` (Python Language Server framework)
- **Validation**: Delegates to `gnn.schema` (same validation as `gnn validate`)
- **Hover**: Parses state-space to show variable metadata at cursor position
- **Fallback**: Graceful degradation when `pygls` is not installed

## File Structure

```
lsp/
├── __init__.py    # Server implementation (226 lines)
├── AGENTS.md      # Agent documentation
├── README.md      # This file
└── SPEC.md        # Module specification
```

## References

- [SPEC.md](SPEC.md) — Module specification
- [AGENTS.md](AGENTS.md) — Agent documentation
