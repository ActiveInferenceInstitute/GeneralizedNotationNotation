# Export Module - Agent Scaffolding

## Module Overview

**Purpose**: Multi-format export generation (JSON, XML, GraphML, GEXF, Pickle) from parsed GNN models

**Pipeline Step**: Step 7: Multi-format export (7_export.py)

**Category**: Data Export / Transformation

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-04-16

---

## Core Functionality

### Primary Responsibilities

1. Export parsed GNN models to multiple formats
2. Generate graph-based representations (GraphML, GEXF)
3. Create portable serializations (JSON, XML, Pickle)
4. Validate export integrity
5. Provide format-specific documentation

### Key Capabilities

- JSON export with schema validation
- XML export with DTD/XSD
- GraphML for network analysis tools
- GEXF for Gephi visualization
- Pickle for Python persistence

---

## API Reference

### Public Functions

#### `process_export(target_dir, output_dir, verbose=False, **kwargs) -> bool`

**Description**: Pipeline entry point (called by `7_export.py`). Loads parsed GNN specs from Step 3 output (`gnn_processing_results.json`) and exports each file to the requested formats. Accepts a `formats` keyword (list of format names).

#### `generate_exports(target_dir, output_dir, verbose=False) -> bool`

**Description**: Standalone export over the `*.md` files directly in `target_dir`; writes to `output_dir/exports/`.

#### `export_model(model_data, output_dir, formats=None) -> Dict[str, Any]`

**Description**: Export one already-parsed model dict to the selected formats; returns a per-format result dictionary.

**Example**:

```python
from export import generate_exports

success = generate_exports(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True,
)
```

---

## Supported Export Formats

### Standard Formats

1. **JSON**: Human-readable, widely compatible
2. **XML**: Schema-validated, industry standard

### Graph Formats

3. **GraphML**: Standard graph format (Cytoscape, yEd)
4. **GEXF**: Gephi visualization format

### Text Formats

5. **Plaintext Summary**: Human-readable model overview
6. **Plaintext DSL**: Round-trip GNN-like text

### Binary Formats

7. **Pickle**: Fast Python serialization

---

## Configuration

### Configuration Options

`process_export` accepts a single keyword:

- `formats` (List[str]): Formats to export (default: `["json", "xml", "graphml", "gexf", "pickle"]`)
  - `"json"`, `"xml"`, `"graphml"`, `"gexf"`, `"pickle"` map to the corresponding exporter
  - `export_model` also supports only these five; anything else is recorded as "Unsupported format". The plaintext formatters (`export_to_plaintext_summary`, `export_to_plaintext_dsl`) exist but are not wired into `export_model` or the pipeline default.

There are no other configuration options.


---

## Dependencies

### Required Dependencies

- `json` - JSON export
- `xml.etree.ElementTree` - XML export
- `pickle` - Pickle serialization

### Optional Dependencies

- `networkx` - Graph format export (recovery: basic XML-based export)

---

## Usage Examples

### Basic Usage

```python
from export import generate_exports

success = generate_exports(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True,
)
```

### Specific Formats

```python
from export import export_model

results = export_model(
    model_data=parsed_data,
    output_dir=Path("output/7_export_output"),
    formats=["json", "graphml", "gexf"],
)
```

---

## Output Specification

`process_export` (pipeline path) writes under the step output dir:

```
output/7_export_output/
├── model_name/
│   ├── model_name.json
│   ├── model_name.xml
│   ├── model_name.graphml
│   ├── model_name.gexf
│   └── model_name_pickle.pkl
├── export_results.json
└── export_summary.json
```

`generate_exports` (standalone) instead writes `{stem}.{json,xml,graphml,gexf,pkl}` plus `export_results.json` under `output_dir/exports/`.

---

## Integration Points

### Pipeline Integration

- **Input**: Receives parsed GNN models from Step 3 (gnn processing)
- **Output**: Generates exports consumed by Step 8 (visualization), Step 11 (render), and Step 20 (website generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies

- **gnn/**: Reads parsed GNN model data for export
- **visualization/**: Provides graph formats for visualization
- **render/**: Provides model data for code generation
- **website/**: Provides export data for website generation

### External Integration

- **Cytoscape**: GraphML format for network analysis
- **Gephi**: GEXF format for graph visualization
- **NetworkX**: Graph format conversion and analysis

### Data Flow

```
3_gnn.py (GNN parsing)
  ↓
7_export.py (Multi-format export)
  ↓
  ├→ 8_visualization.py (Graph visualization)
  ├→ 11_render.py (Code generation)
  ├→ 20_website.py (Website integration)
  └→ output/7_export_output/ (Standalone exports)
```

---

## Testing

### Test Files

- `src/tests/export/test_export_overall.py`

### Test Coverage

Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/export/ \
    --cov=src/export --cov-report=term-missing
```

### Key Test Scenarios

1. Multi-format export generation
2. Format validation and error handling
3. Graph format conversion
---

## MCP Integration

### Tools Registered

- `process_export` — Run the export step over a directory of GNN files
- `export_single_gnn_file` — Export a single GNN file to selected formats
- `list_export_formats` — List supported export formats and descriptions
- `validate_export_format` — Check whether a format name is supported

### MCP File Location

- `src/export/mcp.py` — Tool registrations and MCP wrappers

---

## Troubleshooting

### Common Issues

#### Issue 1: Export fails for specific format

**Symptom**: Export succeeds for some formats but fails for others  
**Cause**: Missing optional dependency (networkx) or format-specific errors  
**Solution**:

- Check that required dependencies are installed: `uv pip install networkx`
- Use `--verbose` flag to see detailed error messages
- Check format-specific requirements in documentation

#### Issue 2: GraphML/GEXF export fails

**Symptom**: Graph formats fail to generate  
**Cause**: Missing networkx dependency or invalid graph structure  
**Solution**:

- Install networkx: `uv pip install networkx`
- Verify GNN model has valid connections section
- Check that graph data is properly structured

#### Issue 3: Large model export

**Symptom**: Export is slow or memory-heavy  
**Cause**: Model too large for a single export operation  
**Solution**:

- Export formats individually instead of all at once
- Process models in smaller batches


### Performance Issues

#### Slow Export Performance

**Symptoms**: Export takes longer than expected  
**Diagnosis**:

```bash
# Enable verbose logging
python src/7_export.py --target-dir input/ --verbose
```

**Solutions**:

- Export only needed formats (don't export all formats if not needed)
- Use pickle format for fastest serialization

```bash
# Enable verbose logging
python src/7_export.py --target-dir input/ --verbose
```

**Solutions**:

- Export only needed formats (don't export all formats if not needed)
- Use pickle format for fastest serialization

---

## Version History

### Current Version: 3.0.0

**Features**:

- Multi-format export (JSON, XML, GraphML, GEXF, Pickle, Plaintext Summary, Plaintext DSL)
- Format validation and error handling
- Graph format conversion via NetworkX
- Export integrity verification
- MCP tool integration

**Known Issues**:

- None currently

### Roadmap

- **Future**: Streaming export for very large models

---

## References

### Related Documentation

- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [GNN Export Guide](../../doc/gnn/integration/gnn_export.md)

### External Resources

- [GraphML Specification](http://graphml.graphdrawing.org/)
- [GEXF Format](https://gexf.net/)
- [NetworkX Documentation](https://networkx.org/)

---

**Last Updated**: 2026-04-16
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
