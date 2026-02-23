# Step 4: Model Registry — Versioning and Lifecycle

## Overview

Implements a centralized model registry for GNN models with versioning, metadata management, and model lifecycle tracking.

## Usage

```bash
python src/4_model_registry.py --target-dir input/gnn_files --output-dir output --verbose
python src/4_model_registry.py --registry-path path/to/registry.json --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/4_model_registry.py` (65 lines) |
| Module | `src/model_registry/` |
| Module function | `process_model_registry()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--registry-path` | `str` | Path to model registry file |

## Output

- **Directory**: `output/4_model_registry_output/`
- Model metadata, versioning information, and registry summary

## Source

- **Script**: [src/4_model_registry.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/4_model_registry.py)
