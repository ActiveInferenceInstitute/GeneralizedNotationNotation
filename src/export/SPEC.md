# Export Module Specification

## Overview

Multi-format export generation from parsed GNN models. Pipeline Step 7.

## Architecture

The export module uses a layered design:

- **`processor.py`** — Orchestrates multi-file and multi-format export workflows (`generate_exports`, `export_single_gnn_file`, `export_model`, `export_gnn_model`). Includes a built-in GNN content parser (`parse_gnn_content`).
- **`formatters.py`** — Format-specific serializers that write parsed GNN data to disk. Each function takes a data dict and an output `Path`, returning `bool` for success.
- **`format_exporters.py`** — Advanced GNN-aware exporters with full section parsing (`_gnn_model_to_dict`), NetworkX graph construction, and logged output. Functions return `Tuple[bool, str]`.
- **`core.py`** — Pipeline integration adapter invoked by `7_export.py`.
- **`utils.py`** — Module introspection (`get_module_info`, `get_supported_formats`).
- **`mcp.py`** — Model Context Protocol tool registrations.

## Export Formats

| Format | Extension | Module | Notes |
|--------|-----------|--------|-------|
| JSON | `.json` | formatters / format_exporters | Human-readable, portable |
| XML | `.xml` | formatters / format_exporters | Hierarchical with minidom pretty-print |
| GraphML | `.graphml` | formatters / format_exporters | Requires NetworkX (optional) |
| GEXF | `.gexf` | formatters / format_exporters | Gephi-compatible, requires NetworkX |
| Pickle | `.pkl` | formatters / format_exporters | Python binary serialization |
| Plaintext Summary | `.txt` | formatters / format_exporters | Human-readable model summary |
| Plaintext DSL | `.dsl` | formatters / format_exporters | Round-trip GNN-like text |

## Key Exports

```python
from export import generate_exports, export_single_gnn_file, export_model
from export import get_supported_formats, get_module_info
```

## Dependencies

- **Required**: `json`, `xml.etree.ElementTree`, `pickle`, `pathlib`
- **Optional**: `networkx` (for GraphML/GEXF graph exports)

## Testing

```bash
uv run python -m pytest src/tests/test_export_integration.py -v
```
