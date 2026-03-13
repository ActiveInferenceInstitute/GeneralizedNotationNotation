# Step 3: GNN — File Discovery and Parsing

## Overview

Discovers, parses, and serializes GNN specification files into multiple output formats. This is the foundational step that most downstream steps depend on.

## Usage

```bash
python src/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/3_gnn.py` (30 lines) |
| Module | `src/gnn/` |
| Processor | `src/gnn/processor.py` |
| Module function | `process_gnn_multi_format()` |
| POMDP extractor | `src/gnn/pomdp_extractor.py` |

This is the **cleanest thin orchestrator** in the pipeline — only 31 lines with a direct import (no try/except alternative).

## Key Capabilities

- Discovers `.md` GNN files in the target directory
- Parses GNN markdown into structured dictionaries
- Extracts state spaces, parameters, connections, and matrices
- Serializes to multiple output formats (JSON, YAML, etc.)
- Handles POMDP-specific extraction via `pomdp_extractor.py`

## Downstream Dependencies

Steps 4, 5, 7, 8, 10, 11, 13, 15, and 16 all depend on Step 3's output.

## Source

- **Script**: [src/3_gnn.py](#placeholder)
