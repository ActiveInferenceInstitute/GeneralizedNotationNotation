# Export Module - Agent Scaffolding

## Module Overview

**Purpose**: Multi-format export generation (JSON, XML, GraphML, GEXF, Pickle) from parsed GNN models

**Pipeline Step**: Step 7: Multi-format export (7_export.py)

**Category**: Data Export / Transformation

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

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

#### `process_export(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main export processing function called by orchestrator (7_export.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for exports
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance (default: None)
- `formats` (List[str]): Formats to export (default: ['json', 'xml', 'graphml', 'gexf', 'pickle'])
- `**kwargs`: Additional options

**Returns**: `True` if export succeeded

**Example**:
```python
from export import process_export

success = process_export(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True,
    formats=['json', 'xml', 'graphml']
)
```

---

## Supported Export Formats

### Standard Formats
1. **JSON**: Human-readable, widely compatible
2. **XML**: Schema-validated, industry standard
3. **YAML**: Configuration-friendly format

### Graph Formats
4. **GraphML**: Standard graph format (Cytoscape, yEd)
5. **GEXF**: Gephi visualization format

### Binary Formats
6. **Pickle**: Fast Python serialization

---

## Configuration

### Configuration Options

#### Export Format Selection
- `export_formats` (List[str]): Formats to export (default: `["json", "xml", "graphml", "gexf", "pickle"]`)
  - `"json"`: JSON export (human-readable)
  - `"xml"`: XML export (schema-validated)
  - `"yaml"`: YAML export (configuration-friendly)
  - `"graphml"`: GraphML format (Cytoscape, yEd)
  - `"gexf"`: GEXF format (Gephi)
  - `"pickle"`: Python pickle serialization

#### Export Options
- `include_metadata` (bool): Include metadata in exports (default: `True`)
- `validate_schema` (bool): Validate XML schema (default: `True`)
- `pretty_print` (bool): Pretty-print JSON/XML (default: `True`)
- `compress` (bool): Compress large exports (default: `False`)

#### Graph Export Configuration
- `graph_layout` (str): Graph layout algorithm (default: `"force"`)
  - Options: `"force"`, `"hierarchical"`, `"circular"`
- `include_weights` (bool): Include edge weights in graphs (default: `True`)
- `node_attributes` (List[str]): Node attributes to include (default: all)

---

## Dependencies

### Required Dependencies
- `json` - JSON export
- `xml.etree.ElementTree` - XML export
- `pickle` - Pickle serialization

### Optional Dependencies
- `yaml` - YAML export (fallback: skip)
- `networkx` - Graph format export (fallback: basic export)

---

## Usage Examples

### Basic Usage
```python
from export import process_export

success = process_export(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    logger=logger
)
```

### Specific Formats
```python
success = process_export(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    logger=logger,
    export_formats=["json", "graphml", "gexf"]
)
```

---

## Output Specification

### Output Products
- `{model}_export.json` - JSON export
- `{model}_export.xml` - XML export
- `{model}_graph.graphml` - GraphML graph
- `{model}_graph.gexf` - GEXF graph
- `{model}_model.pkl` - Pickle serialization
- `export_summary.json` - Export summary

### Output Directory Structure
```
output/7_export_output/
├── model_name_export.json
├── model_name_export.xml
├── model_name_graph.graphml
├── model_name_graph.gexf
├── model_name_model.pkl
└── export_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 61ms
- **Memory**: 28.7 MB
- **Status**: SUCCESS_WITH_WARNINGS
- **Formats Generated**: 5

---

## Error Handling

### Graceful Degradation
- **Format Unavailable**: Skip unavailable format, log warning, continue with others
- **Schema Validation Failure**: Export without validation, log warning
- **Large Model**: Use compression or split exports, provide warnings
- **Invalid GNN Model**: Return structured error, skip model

### Error Categories
1. **Format Errors**: Format not supported or unavailable (fallback: skip format)
2. **Validation Errors**: Schema validation fails (fallback: export without validation)
3. **Serialization Errors**: Cannot serialize model (return error)
4. **File I/O Errors**: Cannot write export files (return error)

### Error Recovery
- **Format Fallback**: Automatically skip unavailable formats
- **Partial Export**: Export what's possible, report failures
- **Resource Cleanup**: Proper cleanup of export resources on errors

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
- `src/tests/test_export_integration.py`

### Test Coverage
- **Current**: 86%
- **Target**: 90%+

---


