# Step 17: Integration — Cross-Module Processing

## Overview

Orchestrates cross-module integration processing for GNN models, ensuring consistency and data flow between pipeline stages.

## Usage

```bash
python src/17_integration.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/17_integration.py` (63 lines) |
| Module | `src/integration/` |
| Processor | `src/integration/processor.py` |
| Module function | `process_integration()` |

## Output

- **Directory**: `output/17_integration_output/`
- Integration reports and cross-module consistency checks

## Source

- **Script**: [src/17_integration.py](../../src/17_integration.py)
