# Step 7: Export — Multi-Format Serialization

## Overview

Generates exports of parsed GNN models in multiple serialization formats.

## Usage

```bash
python src/7_export.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/7_export.py` (55 lines) |
| Module | `src/export/` |
| Processor | `src/export/processor.py` |
| Module function | `process_export()` |

Uses direct import (no try/except handlers) — one of the cleanest orchestrators.

## Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON | `.json` | Standard JSON serialization |
| XML | `.xml` | XML document format |
| GraphML | `.graphml` | Graph Markup Language |
| GEXF | `.gexf` | Graph Exchange XML Format |
| Pickle | `.pkl` | Python binary serialization |

## Output

- **Directory**: `output/7_export_output/`
- Multi-format export files and processing summary

## Source

- **Script**: [src/7_export.py](#placeholder)
