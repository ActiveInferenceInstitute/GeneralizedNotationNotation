# Export Module Specification

## Overview

Multi-format export generation from parsed GNN models. Pipeline Step 7.

## Architecture

The export module uses a layered design:

- **`processor.py`** — Orchestrates multi-file and multi-format export workflows (`process_export`, `generate_exports`, `export_single_gnn_file`, `export_model`, `export_gnn_model`). Includes a built-in GNN content parser (`parse_gnn_content`) and `_gnn_model_to_dict`.
- **`formatters.py`** — Format-specific serializers that write parsed GNN data to disk. Each function takes a data dict and an output `Path`, returning `bool` for success.
- **`format_exporters.py`** — GNN-aware exporters with full section parsing, NetworkX graph construction, and logged output. Functions return `Tuple[bool, str]`.
- **`utils.py`** — Module introspection (`get_module_info`, `get_supported_formats`).
- **`mcp.py`** — Model Context Protocol tool registrations.

## Export Formats

| Format | Extension | Module | Notes |
| XML | `.xml` | formatters / format_exporters | Hierarchical; pretty-printed via `ET.indent` |
| JSON | `.json` | formatters / format_exporters | Human-readable, portable |
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
uv run --extra dev python -m pytest src/tests/export/ -v
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
