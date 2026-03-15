# LSP Module — Specification

## Purpose

Provide Language Server Protocol support for GNN files, enabling real-time validation and in-editor assistance.

## Requirements

1. **Diagnostics**: Publish validation results on `textDocument/didOpen` and `textDocument/didSave`
2. **Hover**: Return variable metadata (name, dimensions, type) on `textDocument/hover`
3. **Graceful fallback**: Function without `pygls` — return `None` from `create_server()`
4. **Validation parity**: Use same validation functions as `gnn.schema` (section validation, state-space parsing, connection parsing)

## Interface

```python
def create_server() -> Optional[LanguageServer]:
    """Create a configured LSP server. Returns None if pygls unavailable."""

def start_server() -> None:
    """Start the LSP server on stdio."""
```

## Constraints

- Server runs over stdio only (no TCP)
- No state persistence between sessions
- Diagnostics sourced from `gnn.schema` — no duplicate validation logic
- Optional dependency: `pygls >= 1.0.0`
