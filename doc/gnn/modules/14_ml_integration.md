# Step 14: ML Integration — Machine Learning Framework Integration

## Overview

Orchestrates ML framework integration for GNN models, connecting parsed specifications with machine learning tools and workflows.

## Usage

```bash
python src/14_ml_integration.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/14_ml_integration.py` (63 lines) |
| Module | `src/ml_integration/` |
| Processor | `src/ml_integration/processor.py` |
| Module function | `process_ml_integration()` |

## Output

- **Directory**: `output/14_ml_integration_output/`
- ML integration reports and configuration summaries

## Source

- **Script**: [src/14_ml_integration.py](../../../src/14_ml_integration.py)
