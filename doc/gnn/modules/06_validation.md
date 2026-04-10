# Step 6: Validation — Semantic Validation and QA

## Overview

Performs semantic validation, performance profiling, and consistency checking on GNN models. Builds on Step 5's type checking with deeper analysis.

## Usage

```bash
python src/6_validation.py --target-dir input/gnn_files --output-dir output --verbose
python src/6_validation.py --strict --profile --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/6_validation.py` (47 lines) |
| Module | `src/validation/` |
| Module function | `process_validation()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--strict` | `bool` | Enable strict validation mode |
| `--profile` | `bool` | Enable performance profiling |

## Output

- **Directory**: `output/6_validation_output/`
- Semantic validation reports, quality scores, performance profiles, and consistency checks

## Source

- **Script**: [src/6_validation.py](../../../src/6_validation.py)
