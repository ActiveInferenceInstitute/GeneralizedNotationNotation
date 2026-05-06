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
- Local script-level worker execution via `execution_workers`
- Optional Ray/Dask dispatch via `distributed` and `backend`

## Key Exports
```python
from execute import process_execute, ExecutionResult
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
