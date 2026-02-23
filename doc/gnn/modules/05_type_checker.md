# Step 5: Type Checker — GNN Type Validation

## Overview

Performs type checking and validation on GNN specification files using the `GNNTypeChecker` class. Supports strict validation mode and resource estimation.

## Usage

```bash
python src/5_type_checker.py --target-dir input/gnn_files --output-dir output --verbose
python src/5_type_checker.py --strict --estimate-resources --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/5_type_checker.py` (75 lines) |
| Module | `src/type_checker/` |
| Processor | `src/type_checker/processor.py` |
| Module class | `GNNTypeChecker` |
| Entry function | `GNNTypeChecker().validate_gnn_files()` |

Uses a `_type_check_dispatch()` wrapper to instantiate `GNNTypeChecker` and call its `validate_gnn_files()` method.

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--strict` | `bool` | Enable strict validation mode |
| `--estimate-resources` | `bool` | Enable resource estimation |

## Output

- **Directory**: `output/5_type_checker_output/`
- Type checking reports, validation results, and resource estimates

## Source

- **Script**: [src/5_type_checker.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/5_type_checker.py)
