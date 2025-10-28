# oxdraw Integration Module

Visual diagram-as-code interface for GNN Active Inference model construction through bidirectional GNN ↔ Mermaid ↔ oxdraw synchronization.

## Overview

The oxdraw integration module enables researchers to:
- Visually construct and edit Active Inference models using oxdraw's interactive editor
- Convert GNN specifications to Mermaid flowchart format
- Parse edited Mermaid diagrams back to GNN format
- Preserve Active Inference ontology mappings through metadata embedding
- Validate models through the complete GNN pipeline

## Features

✅ **Bidirectional Conversion**: GNN ↔ Mermaid with full fidelity  
✅ **Interactive Editing**: Launch oxdraw editor for visual model construction  
✅ **Metadata Preservation**: Ontology terms and dimensions embedded in Mermaid comments  
✅ **Headless Mode**: Batch conversion without GUI for automation  
✅ **Node Shape Mapping**: Automatic shapes based on variable types  
✅ **Edge Style Mapping**: Visual distinction of connection types  
✅ **MCP Integration**: Model Context Protocol tools for external access  

## Installation

### Required Dependencies
```bash
# Install GNN pipeline dependencies
pip install -r requirements.txt
```

### Optional: oxdraw CLI
For interactive visual editing, install oxdraw via Cargo:
```bash
# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install oxdraw
cargo install oxdraw

# Verify installation
oxdraw --version
```

**Note**: The module works in headless mode without oxdraw CLI (conversion only).

## Quick Start

### Basic Usage: GNN to Mermaid Conversion

```python
from pathlib import Path
from oxdraw import convert_gnn_file_to_mermaid

# Convert GNN file to Mermaid
convert_gnn_file_to_mermaid(
    gnn_file_path=Path("input/actinf_pomdp_agent.md"),
    output_path=Path("output/actinf_pomdp_agent.mmd")
)
```

### Interactive Usage: Launch oxdraw Editor

```python
from pathlib import Path
from oxdraw import launch_oxdraw_editor

# Launch editor for visual editing
launch_oxdraw_editor(
    mermaid_file=Path("output/actinf_pomdp_agent.mmd"),
    port=5151,
    host="127.0.0.1"
)

# Editor opens at http://127.0.0.1:5151
```

### Round-Trip: GNN → oxdraw → GNN

```python
from pathlib import Path
from oxdraw import (
    convert_gnn_file_to_mermaid,
    convert_mermaid_file_to_gnn
)

# Step 1: Convert GNN to Mermaid
gnn_file = Path("input/model.md")
mermaid_file = Path("output/model.mmd")
convert_gnn_file_to_mermaid(gnn_file, mermaid_file)

# Step 2: Edit visually in oxdraw (manual step)
# ... user edits diagram ...

# Step 3: Convert back to GNN
edited_gnn = Path("output/model_edited.md")
gnn_model = convert_mermaid_file_to_gnn(mermaid_file, edited_gnn)

print(f"Variables: {len(gnn_model['variables'])}")
print(f"Connections: {len(gnn_model['connections'])}")
```

## Pipeline Integration

### Run as Pipeline Step (Step 24)

```bash
# Headless mode (no GUI, fast)
python3 src/24_oxdraw.py \
    --target-dir input/gnn_files \
    --output-dir output \
    --mode headless \
    --verbose

# Interactive mode (launches editor)
python3 src/24_oxdraw.py \
    --target-dir input/gnn_files \
    --output-dir output \
    --mode interactive \
    --launch-editor \
    --port 5151 \
    --verbose
```

### Integration with main.py

The oxdraw step can be added to the full pipeline:

```bash
# Run full pipeline including oxdraw
python3 src/main.py --only-steps "3,24" --verbose
```

## Node Shape Mapping

GNN variable types are automatically mapped to Mermaid shapes:

| Variable Type | Shape | Syntax | Example |
|--------------|-------|--------|---------|
| Matrix (A, B) | Rectangle | `[A]` | `A[A<br/>3x3<br/>float]` |
| Vector (C, D, E) | Rounded | `(C)` | `C(C<br/>3<br/>float)` |
| State (s, s_prime) | Stadium | `([s])` | `s([s<br/>3x1<br/>float])` |
| Observation (o) | Circle | `((o))` | `o((o<br/>3x1<br/>int))` |
| Action (u) | Hexagon | `{{u}}` | `u{{u<br/>1<br/>int}}` |
| Policy (π) | Diamond | `{π}` | `π{π<br/>3<br/>float}` |
| Free Energy (F, G) | Trapezoid | `[/F\]` | `F[/F<br/>float\]` |

## Edge Style Mapping

GNN connection symbols are mapped to Mermaid edge styles:

| GNN Symbol | Connection Type | Mermaid Style | Visual |
|-----------|-----------------|---------------|--------|
| `>` | Generative | `==>` | Thick arrow |
| `-` | Inference | `-.->` | Dashed arrow |
| `*` | Modulation | `-..->` | Dotted arrow |
| `~` | Weak Coupling | `-->` | Normal arrow |

## API Reference

### Core Functions

#### `process_oxdraw(target_dir, output_dir, logger, **kwargs)`
Main processing function for pipeline integration.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory
- `logger` (Logger): Logger instance
- `mode` (str): "interactive" or "headless" (default: headless)
- `launch_editor` (bool): Launch oxdraw editor
- `port` (int): Editor port (default: 5151)

**Returns**: `bool` - Success status

#### `gnn_to_mermaid(gnn_model, include_metadata=True)`
Convert parsed GNN model to Mermaid flowchart.

**Parameters**:
- `gnn_model` (Dict): Parsed GNN model dictionary
- `include_metadata` (bool): Include metadata for bidirectional sync

**Returns**: `str` - Mermaid diagram content

#### `mermaid_to_gnn(mermaid_content, validate_ontology=False)`
Parse Mermaid diagram back to GNN model structure.

**Parameters**:
- `mermaid_content` (str): Mermaid diagram content
- `validate_ontology` (bool): Validate ontology terms

**Returns**: `Dict` - GNN model dictionary

### File Conversion Functions

#### `convert_gnn_file_to_mermaid(gnn_file_path, output_path=None)`
Convert GNN file to Mermaid file.

#### `convert_mermaid_file_to_gnn(mermaid_file_path, output_path=None)`
Convert Mermaid file to GNN file.

### Utility Functions

#### `check_oxdraw_installed()`
Check if oxdraw CLI is available.

**Returns**: `bool`

#### `launch_oxdraw_editor(mermaid_file, port=5151, host="127.0.0.1")`
Launch interactive oxdraw editor.

**Returns**: `bool` - Success status

## Testing

### Run All Tests

```bash
# Run oxdraw integration tests
pytest src/tests/test_oxdraw_integration.py -v

# Run converter tests
pytest src/tests/test_mermaid_converter.py -v

# Run parser tests
pytest src/tests/test_mermaid_parser.py -v

# Run all oxdraw tests
pytest src/tests/test_*oxdraw*.py src/tests/test_mermaid*.py -v
```

### Test Coverage

```bash
# Check test coverage
pytest --cov=src/oxdraw --cov-report=term-missing
```

## Examples

See `doc/oxdraw/gnn_oxdraw.md` for comprehensive examples including:
- Converting `actinf_pomdp_agent.md` to Mermaid
- Visual editing workflows
- Round-trip validation
- Integration with GNN pipeline steps

## Architecture

The module follows the GNN pipeline's **thin orchestrator pattern**:

```
src/
├── 24_oxdraw.py                 # Thin orchestrator
├── oxdraw/                      # Module implementation
│   ├── __init__.py             # Public API
│   ├── processor.py            # Main processing logic
│   ├── mermaid_converter.py    # GNN → Mermaid
│   ├── mermaid_parser.py       # Mermaid → GNN
│   ├── utils.py                # Helper functions
│   ├── mcp.py                  # MCP tool registration
│   └── AGENTS.md               # Comprehensive documentation
└── tests/
    ├── test_oxdraw_integration.py
    ├── test_mermaid_converter.py
    └── test_mermaid_parser.py
```

## Performance

- **GNN → Mermaid**: 10-50ms per file
- **Mermaid → GNN**: 20-100ms per file
- **oxdraw Launch**: 1-2s startup time
- **Memory Usage**: <10MB (excluding oxdraw process)
- **Scalability**: Tested up to 100 variables, 200 connections

## Troubleshooting

### "oxdraw CLI not found"
**Solution**: Install via `cargo install oxdraw` or use headless mode

### "Metadata not preserved"
**Solution**: Ensure `include_metadata=True` in conversion functions

### "Invalid ontology terms"
**Solution**: Check `input/ontology_terms.json` or disable validation

### "Mermaid syntax errors"
**Diagnostic**: Use `validate_mermaid_syntax()` from `oxdraw.utils`

## MCP Integration

The module registers MCP tools for external access:

- `oxdraw.convert_to_mermaid` - Convert GNN to Mermaid
- `oxdraw.convert_from_mermaid` - Convert Mermaid to GNN
- `oxdraw.launch_editor` - Launch interactive editor
- `oxdraw.check_installation` - Check CLI availability
- `oxdraw.get_info` - Get module information

## Contributing

Follow GNN pipeline development guidelines:
- Maintain thin orchestrator pattern
- Comprehensive test coverage (>90%)
- Type hints for all public functions
- Docstrings with examples
- No mock implementations

## References

- [oxdraw GitHub](https://github.com/RohanAdwankar/oxdraw)
- [oxdraw Technical Overview](../../doc/oxdraw/oxdraw.md)
- [GNN-oxdraw Integration Guide](../../doc/oxdraw/gnn_oxdraw.md)
- [GNN Pipeline Documentation](../AGENTS.md)
- [Mermaid Documentation](https://mermaid.js.org/)

## License

MIT License - See main repository LICENSE file

---

**Version**: 1.0.0  
**Last Updated**: October 28, 2025  
**Status**: ✅ Ready for Testing

