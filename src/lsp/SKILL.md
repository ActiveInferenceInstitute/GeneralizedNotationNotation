# Core Skill: `lsp_server`

**Function**: Language Server Protocol implementation providing real-time GNN file diagnostics and hover information for LSP-compatible editors.

## Example Flow

```bash
# Start the GNN Language Server via CLI
gnn lsp

# Or directly via Python module
python -m src.lsp
```

## Programmatic Usage

```python
from lsp import create_server, start_server

# Create and configure the LSP server
server = create_server()

# Start on stdio (standard usage for editor integration)
start_server()
```

## Features

- **Diagnostic Publishing**: Validates GNN sections, state-space variables, and connections on file open and save
- **Hover Information**: Displays variable name, dimensions, type, and default values at cursor position
- **Error Extraction**: Maps parse errors to line numbers for inline display in editors
- **Graceful Fallback**: Works without `pygls` by providing a clear installation message
- **VS Code Integration**: Compatible with any LSP client via stdio transport
