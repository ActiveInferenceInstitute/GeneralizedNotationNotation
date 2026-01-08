# Execute Module Specification

## Overview
GNN model execution across multiple frameworks.

## Components

### Core
- `processor.py` - Execution processor (1380 lines)

### Framework Runners
- `jax/` - JAX execution
- `pymdp/` - PyMDP execution
- `numpy/` - NumPy execution

## Execution Modes
- Single model execution
- Batch execution
- Framework-specific execution

## Key Exports
```python
from execute import process_execute, ExecutionResult
```
