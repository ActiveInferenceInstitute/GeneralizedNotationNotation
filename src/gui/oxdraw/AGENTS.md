# oxdraw Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: Visual diagram-as-code interface for Active Inference model construction through bidirectional GNN ↔ Mermaid ↔ oxdraw synchronization

**Pipeline Step**: Step 22: GUI Processing - oxdraw option (22_gui.py)

**Parent Module**: gui (Interactive GNN Constructors)

**Category**: Interactive Visualization / Model Construction

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN Active Inference models to Mermaid flowchart format
2. Parse Mermaid diagrams edited in oxdraw back to GNN format
3. Launch interactive oxdraw editor for visual model construction
4. Preserve Active Inference ontology mappings through metadata embedding
5. Validate bidirectional conversions for semantic consistency

### Key Capabilities
- GNN → Mermaid conversion with embedded metadata
- Mermaid → GNN parsing with visual edit preservation
- Interactive visual editing through oxdraw CLI integration
- Headless batch conversion for automation
- Ontology term preservation and validation
- Connection topology validation

---

## API Reference

### Public Functions

#### `process_oxdraw(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main processing function for oxdraw integration

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for Mermaid files and results
- `logger` (Logger): Logger instance for progress reporting
- `mode` (str): "interactive" or "headless" (default: headless)
- `auto_convert` (bool): Auto-convert GNN files to Mermaid
- `validate_on_save` (bool): Validate Mermaid → GNN conversions
- `launch_editor` (bool): Launch oxdraw editor (interactive mode only)
- `port` (int): oxdraw server port (default: 5151)
- `host` (str): oxdraw server host (default: 127.0.0.1)

**Returns**: `True` if processing succeeded

**Example**:
```python
from oxdraw.processor import process_oxdraw

success = process_oxdraw(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/24_oxdraw_output"),
    logger=logger,
    mode="interactive",
    launch_editor=True
)
```

#### `gnn_to_mermaid(gnn_model, include_metadata=True) -> str`
**Description**: Convert parsed GNN model to Mermaid flowchart format

**Parameters**:
- `gnn_model` (Dict): Parsed GNN model dictionary
- `include_metadata` (bool): Include GNN metadata in comments

**Returns**: Mermaid flowchart string

**Example**:
```python
from gnn.processor import parse_gnn_file
from oxdraw.mermaid_converter import gnn_to_mermaid

gnn_model = parse_gnn_file("model.md")
mermaid_diagram = gnn_to_mermaid(gnn_model)
```

#### `mermaid_to_gnn(mermaid_content, validate_ontology=False) -> Dict`
**Description**: Parse Mermaid flowchart back to GNN model structure

**Parameters**:
- `mermaid_content` (str): Mermaid diagram content
- `validate_ontology` (bool): Validate ontology term mappings

**Returns**: GNN model dictionary

**Example**:
```python
from oxdraw.mermaid_parser import mermaid_to_gnn

mermaid_content = Path("diagram.mmd").read_text()
gnn_model = mermaid_to_gnn(mermaid_content, validate_ontology=True)
```

#### `convert_gnn_file_to_mermaid(gnn_file_path, output_path=None) -> str`
**Description**: Convert GNN file to Mermaid format for oxdraw

**Example**:
```python
from oxdraw import convert_gnn_file_to_mermaid

convert_gnn_file_to_mermaid(
    Path("input/actinf_pomdp_agent.md"),
    Path("output/actinf_pomdp_agent.mmd")
)
```

#### `convert_mermaid_file_to_gnn(mermaid_file_path, output_path=None) -> Dict`
**Description**: Convert Mermaid file back to GNN format

**Example**:
```python
from oxdraw import convert_mermaid_file_to_gnn

gnn_model = convert_mermaid_file_to_gnn(
    Path("edited_diagram.mmd"),
    Path("output/edited_model.md")
)
```

---

## Dependencies

### Required Dependencies
- `pathlib` - File path operations
- `json` - Metadata serialization
- `re` - Pattern matching for parsing

### Optional Dependencies
- `oxdraw` (Rust CLI) - Interactive visual editor (fallback: headless mode only)
- `ontology.processor` - Ontology validation (fallback: skip validation)

### Internal Dependencies
- `gnn.processor` - GNN file parsing and discovery
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `OXDRAW_DEFAULT_PORT` - Default port for oxdraw server (default: 5151)
- `OXDRAW_DEFAULT_HOST` - Default host address (default: 127.0.0.1)
- `OXDRAW_MODE` - Default processing mode (default: headless)

### Default Settings
```python
DEFAULT_OXDRAW_SETTINGS = {
    'mode': 'headless',
    'port': 5151,
    'host': '127.0.0.1',
    'auto_convert': True,
    'validate_on_save': True,
    'include_metadata': True,
    'include_styling': True
}
```

---

## Node Shape Mapping

### GNN Variable Types → Mermaid Shapes

| Variable Type | Mermaid Shape | Syntax | Example |
|--------------|---------------|---------|---------|
| Matrix | Rectangle | `[A]` | A[A<br/>3x3<br/>float] |
| Vector | Rounded | `(C)` | C(C<br/>3<br/>float) |
| State | Stadium | `([s])` | s([s<br/>3x1<br/>float]) |
| Observation | Circle | `((o))` | o((o<br/>3x1<br/>int)) |
| Action | Hexagon | `{{u}}` | u{{u<br/>1<br/>int}} |
| Policy | Diamond | `{π}` | π{π<br/>3<br/>float} |
| Free Energy | Trapezoid | `[/F\]` | F[/F<br/>float\] |

---

## Edge Style Mapping

### GNN Connection Symbols → Mermaid Styles

| GNN Symbol | Connection Type | Mermaid Style | Example |
|-----------|-----------------|---------------|---------|
| `>` | Generative | `==>` | D ==> s |
| `-` | Inference | `-.->` | s -.-> A |
| `*` | Modulation | `-..->` | γ -..-> F |
| `~` | Weak Coupling | `-->` | x --> y |

---

## Usage Examples

### Basic Usage: Headless Conversion

```python
from pathlib import Path
from oxdraw import process_oxdraw
import logging

logger = logging.getLogger(__name__)

# Convert GNN files to Mermaid in headless mode
success = process_oxdraw(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/oxdraw_output"),
    logger=logger,
    mode="headless",
    auto_convert=True
)

print(f"Conversion {'succeeded' if success else 'failed'}")
```

### Interactive Usage: Launch Editor

```python
# Launch oxdraw editor for visual editing
success = process_oxdraw(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/oxdraw_output"),
    logger=logger,
    mode="interactive",
    launch_editor=True,
    port=5151,
    host="127.0.0.1"
)

# Editor opens at http://127.0.0.1:5151
# User edits diagram visually, saves, and closes
```

### Pipeline Integration

```python
# Run as part of GNN pipeline (Step 22 - GUI module)
import subprocess

subprocess.run([
    "python3", "src/22_gui.py",
    "--target-dir", "input/gnn_files",
    "--output-dir", "output",
    "--gui-types", "oxdraw",
    "--headless",
    "--verbose"
])
```

---

## Input/Output Specification

### Input Requirements
- **GNN Files**: `.md` files with valid GNN syntax
- **Mermaid Files**: `.mmd` files with flowchart directive (for conversion back)
- **Prerequisites**: Step 3 (GNN parsing) completion recommended but not required

### Output Products
- `{model_name}.mmd` - Mermaid flowchart files
- `{model_name}_from_mermaid.md` - Regenerated GNN files
- `oxdraw_processing_results.json` - Processing summary

### Output Directory Structure
```
output/24_oxdraw_output/
├── actinf_pomdp_agent.mmd
├── actinf_pomdp_agent_from_mermaid.md
├── model2.mmd
├── model2_from_mermaid.md
└── oxdraw_processing_results.json
```

---

## Workflow Example

### Complete Workflow: GNN → oxdraw → GNN

```python
from pathlib import Path
from oxdraw import (
    convert_gnn_file_to_mermaid,
    launch_oxdraw_editor,
    convert_mermaid_file_to_gnn
)

# Step 1: Convert GNN to Mermaid
gnn_file = Path("input/actinf_pomdp_agent.md")
mermaid_file = Path("output/actinf_pomdp_agent.mmd")

convert_gnn_file_to_mermaid(gnn_file, mermaid_file)
print(f"✅ Created Mermaid file: {mermaid_file}")

# Step 2: Launch oxdraw editor (interactive)
launch_oxdraw_editor(mermaid_file, port=5151)
print("🎨 Edit your model at http://127.0.0.1:5151")
print("💾 Save and close when done...")

# Step 3: Convert edited Mermaid back to GNN
edited_gnn = Path("output/actinf_edited.md")
gnn_model = convert_mermaid_file_to_gnn(mermaid_file, edited_gnn)

print(f"✅ Created GNN file: {edited_gnn}")
print(f"   Variables: {len(gnn_model['variables'])}")
print(f"   Connections: {len(gnn_model['connections'])}")
```

---

## Error Handling

### Error Categories
1. **oxdraw Not Installed**: Falls back to headless mode
2. **Invalid GNN Syntax**: Logs error, skips file
3. **Malformed Mermaid**: Logs error with line number
4. **Ontology Validation**: Warnings for invalid terms
5. **File I/O Errors**: Graceful error handling with context

### Fallback Strategies
- **No oxdraw CLI**: Headless conversion only (no interactive editing)
- **Invalid Metadata**: Use visual structure only
- **Missing Ontology**: Skip ontology validation
- **Parser Errors**: Generate minimal valid output

---

## Integration Points

### Orchestrated By
- **Script**: `22_gui.py` (Step 22)
- **Parent Module**: `gui` (Interactive GNN Constructors)
- **Function**: `oxdraw_gui()` → `process_oxdraw()`

### Imports From
- `gnn.processor` - GNN file parsing
- `ontology.processor` - Ontology validation
- `utils.pipeline_template` - Standardized processing

### Imported By
- `gui.__init__.py` - GUI module aggregator
- `tests.test_oxdraw_integration.py` - Integration tests
- `main.py` - Pipeline orchestration via GUI module

### Data Flow
```
GNN Files → parse_gnn_file() → gnn_to_mermaid() → Mermaid Files
                ↓                                          ↓
            Variables                                 oxdraw Editor
            Connections                                    ↓
            Ontology                                  Visual Edits
                ↑                                          ↓
            Merged Model ← mermaid_to_gnn() ← Edited Mermaid
```

---

## Testing

### Test Files
- `src/tests/test_oxdraw_integration.py` - Integration tests
- `src/tests/test_mermaid_converter.py` - Converter unit tests
- `src/tests/test_mermaid_parser.py` - Parser unit tests

### Test Coverage
- **Current**: New module (comprehensive tests included)
- **Target**: 90%+

### Key Test Scenarios
1. GNN → Mermaid conversion with metadata
2. Mermaid → GNN parsing with visual edits
3. Round-trip conversion (GNN → Mermaid → GNN)
4. Ontology preservation and validation
5. Error handling for malformed inputs
6. Node shape inference from variable types
7. Edge style mapping from connection symbols
8. Metadata extraction and serialization

---

## MCP Integration

### Tools Registered
- `oxdraw.convert_to_mermaid` - Convert GNN to Mermaid
- `oxdraw.convert_from_mermaid` - Convert Mermaid to GNN
- `oxdraw.launch_editor` - Launch interactive editor
- `oxdraw.check_installation` - Check oxdraw CLI availability
- `oxdraw.get_info` - Get module information

### Tool Endpoints
```python
@mcp_tool("oxdraw.convert_to_mermaid")
def convert_to_mermaid_tool(gnn_file_path: str, output_path: str = None):
    """Convert GNN file to Mermaid format"""
    return convert_gnn_file_to_mermaid(Path(gnn_file_path), Path(output_path))
```

---

## Performance Characteristics

### Expected Performance
- **GNN → Mermaid**: 10-50ms per file
- **Mermaid → GNN**: 20-100ms per file
- **oxdraw Launch**: 1-2s startup time
- **Memory**: <10MB (excluding oxdraw process)

### Scalability
- **Model Size**: Tested up to 100 variables, 200 connections
- **oxdraw Limit**: ~500 nodes (practical visual limit)
- **Batch Processing**: Linear scaling with file count

---

## Troubleshooting

### Common Issues

#### Issue 1: "oxdraw CLI not found"
**Symptom**: Warning about missing oxdraw CLI

**Solution**:
```bash
# Install oxdraw via Cargo
cargo install oxdraw

# Verify installation
oxdraw --version
```

**Fallback**: Module works in headless mode without oxdraw CLI

#### Issue 2: "Metadata not preserved"
**Symptom**: Visual edits lost after conversion

**Cause**: Metadata embedding disabled

**Solution**: Ensure `include_metadata=True` in conversion

#### Issue 3: "Invalid ontology terms"
**Symptom**: Ontology validation errors

**Solution**: 
- Check ontology terms in `src/ontology/act_inf_ontology_terms.json`
- Disable validation with `validate_ontology=False`

#### Issue 4: "Mermaid syntax errors"
**Symptom**: Parser fails on Mermaid files

**Diagnostic**:
```python
from oxdraw.utils import validate_mermaid_syntax

is_valid, errors = validate_mermaid_syntax(mermaid_content)
for error in errors:
    print(f"❌ {error}")
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Bidirectional GNN ↔ Mermaid conversion
- Interactive visual editing via oxdraw
- Ontology preservation and validation
- MCP tool integration
- Comprehensive test coverage

**Known Limitations**:
- Requires manual oxdraw CLI installation
- No real-time sync (save and reload required)
- Limited to flowchart diagrams (no sequence/class diagrams)

### Roadmap
- **1.1.0**: Real-time WebSocket sync with oxdraw
- **1.2.0**: Multi-model hierarchical editing
- **1.3.0**: Custom Active Inference shape library

---

## References

### Related Documentation
- [oxdraw Technical Overview](./README.md)
- [GNN-oxdraw Integration Guide](./README.md)
- [GNN Parser](../../../doc/gnn/AGENTS.md)
- [Ontology Module](../../ontology/AGENTS.md)

### External Resources
- [oxdraw GitHub](https://github.com/RohanAdwankar/oxdraw)
- [Mermaid Documentation](https://mermaid.js.org/)
- [Active Inference Ontology](../../src/ontology/act_inf_ontology_terms.json)

---

**Last Updated**: 2026-01-07  
**Maintainer**: GNN Pipeline Team  
**Status**: ✅ Ready for Testing

