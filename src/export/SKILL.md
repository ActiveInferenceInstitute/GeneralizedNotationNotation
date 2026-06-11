---
name: gnn-export
description: GNN multi-format export generation. Use when exporting GNN models to JSON, XML, GraphML, GEXF, Pickle, or other interchange formats for use in external tools and frameworks.
---

# GNN Export (Step 7)

## Purpose

Exports parsed GNN models to multiple interchange formats for interoperability with external tools, visualization platforms, and analysis frameworks.

## Key Commands

```bash
# Run export
python src/7_export.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 7 --verbose
```

## Supported Formats

| Format | Extension | Use Case |
| -------- | ----------- | ---------- |
| **JSON** | `.json` | General interchange, web tools |
| **XML** | `.xml` | Enterprise integration |
| **GraphML** | `.graphml` | Graph analysis tools (Gephi, yEd) |
| **GEXF** | `.gexf` | Dynamic graph visualization |
| **Pickle** | `.pkl` | Python-native serialization |

## API

```python
from export import (
    export_model, generate_exports, export_single_gnn_file,
    Exporter, MultiFormatExporter, get_supported_formats,
    export_to_json, export_to_xml, export_to_graphml, export_to_gexf,
    process_export
)

# Export single file to all formats
export_single_gnn_file("model.md", "output/")

# Use the Exporter class
exporter = Exporter()
result = exporter.export_gnn_model(gnn_content, "json")

# Multi-format export
mf_exporter = MultiFormatExporter()
results = mf_exporter.export_to_multiple_formats(content, ["json", "xml", "graphml"])

# Query supported formats
formats = get_supported_formats()

# Run full export step (used by pipeline)
process_export(target_dir, output_dir, verbose=True)
```

## Key Exports

- `export_model` / `export_single_gnn_file` — export a single model
- `generate_exports` — batch export for a directory
- `Exporter` / `MultiFormatExporter` — class-based export interfaces
- `export_to_json`, `export_to_xml`, `export_to_graphml`, `export_to_gexf` — format-specific
- `process_export` — main pipeline processing function

## Output

- Exported files in `output/7_export_output/`
- One file per model per format
- Export summary manifest


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `export_single_gnn_file`
- `list_export_formats`
- `process_export`
- `validate_export_format`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
