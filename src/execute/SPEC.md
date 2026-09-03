# Execute Module Specification

## Overview
GNN model execution across multiple frameworks.

## Components

### Core
- `processor.py` - Execution processor (Step 12 entry point, framework parsing, script discovery, pre-flight skips)

### Framework Runners
- `executor.py` - `GNNExecutor` plus the `ExecutorFrameworkSpec` registry (pymdp, rxinfer, discopy, activeinference_jl, jax, numpyro, pytorch)
- `jax/` - JAX execution
- `pymdp/` - PyMDP execution
- `stan/` - Stan execution (cmdstanpy driver runner; skips when CmdStan is absent)
- `numpyro/` - NumPyro execution

## Execution Modes
- Single model execution
- Batch execution
- Framework-specific execution
- Local script-level worker execution via `execution_workers`
- Optional Ray/Dask dispatch via `distributed` and `backend`

## Key Exports
```python
from execute import (
    process_execute,
    execute_script_safely,
    execute_simulation_from_gnn,
    validate_execution_environment,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
