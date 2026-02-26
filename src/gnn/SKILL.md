---
name: gnn-parsing
description: GNN file discovery, parsing, and multi-format serialization. Use when reading GNN model files, parsing StateSpaceBlock definitions, extracting connections, validating GNN syntax, or converting between GNN formats.
---

# GNN Parsing (Step 3)

## Purpose

Discovers, parses, and validates GNN model files. Extracts structured data from Markdown-based GNN specifications including state spaces, connections, parameterizations, and ontology annotations.

## Key Commands

```bash
# Parse all GNN files in a directory
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 3 --verbose
```

## GNN File Sections

| Section | Purpose | Example |
| --------- | --------- | --------- |
| `GNNSection` | Model type declaration | `ActInfPOMDP` |
| `ModelName` | Model identifier | `T-Maze Agent` |
| `StateSpaceBlock` | Matrix/vector definitions | `A[3,3,type=float]` |
| `Connections` | Directed (`>`) and undirected (`-`) edges | `D>s`, `s-A` |
| `InitialParameterization` | Default values | `A={(0.9,0.05,0.05)}` |
| `ActInfOntologyAnnotation` | Semantic labels | `A=LikelihoodMatrix` |

## API

```python
from gnn import discover_gnn_files, parse_gnn_file, process_gnn_directory

# Discover GNN files in a directory
files = discover_gnn_files("input/gnn_files/")

# Parse a single file
model = parse_gnn_file("input/gnn_files/my_model.md")

# Process entire directory (used by pipeline)
results = process_gnn_directory("input/gnn_files/", "output/")

# Multi-format processing
from gnn import process_gnn_multi_format
results = process_gnn_multi_format("input/gnn_files/", "output/")

# Formal parsing and validation
from gnn import validate_gnn, parse_gnn_formal, GNNFormalParser
is_valid, errors = validate_gnn(content)
```

## Key Exports

- `discover_gnn_files` — find `.md` GNN files in a directory tree
- `parse_gnn_file` — parse a single GNN file into structured data
- `process_gnn_directory` — process all files in a directory
- `validate_gnn_structure` — structural validation of a parsed model
- `validate_gnn` — content-level syntax validation
- `GNNFormalParser`, `ParsedGNN`, `ParsedGNNFormal` — parser classes

## Output

Parsed models are consumed by downstream steps:

- **Step 5** (Type Check): validates matrix dimensions
- **Steps 7–8** (Export/Viz): generates outputs
- **Step 11** (Render): generates simulation code


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_gnn_documentation`
- `validate_gnn_content`
- `parse_gnn_content`
- `process_gnn_directory`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [../../doc/gnn/gnn_syntax.md](../../doc/gnn/gnn_syntax.md) — Complete syntax reference
- [../../doc/gnn/gnn_examples_doc.md](../../doc/gnn/gnn_examples_doc.md) — Example models


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
