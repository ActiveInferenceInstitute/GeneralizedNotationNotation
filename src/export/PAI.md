# Export Module - PAI Context

## Quick Reference

**Purpose:** Export parsed GNN models to multiple formats (JSON, XML, GraphML, GEXF, Pickle, Plaintext Summary, Plaintext DSL).

**When to use this module:**

- Export GNN models to portable formats
- Generate graph representations for network analysis
- Create human-readable model summaries
- Serialize models for Python persistence

## Common Operations

```python
# Batch export all GNN files in a directory
from export import generate_exports
success = generate_exports(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True
)

# Export a single GNN file
from export import export_single_gnn_file
results = export_single_gnn_file(
    gnn_file=Path("input/gnn_files/model.md"),
    exports_dir=Path("output/7_export_output/exports")
)

# Export parsed model data to specific formats
from export import export_model
results = export_model(
    model_data=parsed_data,
    output_dir=Path("output/exports"),
    formats=["json", "xml", "graphml"]
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn, validation | Parsed GNN models |
| **Output** | visualization, render, website | JSON, XML, GraphML, GEXF, Pickle, TXT, DSL files |

## Key Files

- `processor.py` - Export orchestration (`generate_exports`, `export_model`, `export_gnn_model`)
- `formatters.py` - Format-specific serializers (JSON, XML, GraphML, GEXF, Pickle, Plaintext)
- `format_exporters.py` - Advanced GNN-aware exporters with NetworkX graph construction
- `core.py` - Pipeline integration adapter
- `utils.py` - Module introspection and format listing
- `mcp.py` - MCP tool registrations
- `__init__.py` - Public API exports

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | .json | Portable, human-readable interchange |
| XML | .xml | Hierarchical, schema-validated |
| GraphML | .graphml | Network analysis (Cytoscape, yEd) |
| GEXF | .gexf | Graph visualization (Gephi) |
| Pickle | .pkl | Python binary serialization |
| Plaintext Summary | .txt | Human-readable overview |
| Plaintext DSL | .dsl | Round-trip GNN-like text |

## Tips for AI Assistants

1. **Step 7:** Export is Step 7 of the pipeline
2. **7 Formats:** JSON, XML, GraphML, GEXF, Pickle, TXT, DSL
3. **Output Location:** `output/7_export_output/`
4. **NetworkX:** GraphML and GEXF require NetworkX (optional dependency)
5. **Preservation:** Maintains all model metadata across formats

---

**Version:** 1.1.3 | **Step:** 7 (Export)
