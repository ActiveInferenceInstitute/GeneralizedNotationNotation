# NumPyro Execution Sub-module

## Overview

Discovers and runs NumPyro-generated POMDP scripts via subprocess. Includes dependency checking, syntax validation, log persistence, and execution timing.

## Architecture

```
numpyro/
├── __init__.py           # Package exports
└── numpyro_runner.py     # NumPyro script discovery and execution (224 lines)
```

## Key Functions

- **`run_numpyro_scripts(render_dir, output_dir)`** — Discovers and executes all NumPyro scripts in the render output directory.
- **Dependency validation** — Checks for `numpyro` and `jax` availability before execution.
- **Syntax checking** — Pre-validates Python syntax before subprocess execution.
- **Log persistence** — Captures stdout/stderr to log files alongside results.
- **Timing** — Records wall-clock execution time for each script.

## Dependencies

- `numpyro`, `jax` (runtime, checked before execution)

## Parent Module

See [execute/AGENTS.md](../AGENTS.md) for the overall execution architecture.

**Version**: 1.6.0
