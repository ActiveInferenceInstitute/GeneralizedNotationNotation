# ngc-learn Execution Sub-module

## Overview

Discovers and runs ngc-learn-generated POMDP scripts via subprocess. Includes dependency checking, syntax validation, log persistence, and execution timing. ngc-learn is an optional extra (`uv sync --extra ngclearn`); when the runtime is absent the backend is reported skipped, never failed.

## Architecture

```
ngclearn/
├── __init__.py           # Package exports
└── ngclearn_runner.py    # ngc-learn script discovery and execution
```

## Key Functions

- **`run_ngclearn_scripts(render_dir, output_dir)`** — Discovers and executes all ngc-learn scripts in the render output directory.
- **Dependency validation** — Checks for `ngcsimlib`, `ngclearn`, `jax`, and `numpy` availability before execution.
- **Syntax checking** — Pre-validates Python syntax before subprocess execution.
- **Log persistence** — Captures stdout/stderr to log files alongside results.
- **Timing** — Records wall-clock execution time for each script.

## Environment

Rendered scripts receive `NGCLEARN_OUTPUT_DIR` pointing at the Step 12 simulation-data directory; results land in `<model>/ngclearn/simulation_data/simulation_results.json`.

## Dependencies

- `ngcsimlib`, `ngclearn`, `jax`, `numpy` (runtime, checked before execution; install with `uv sync --extra ngclearn`)

## Parent Module

See [execute/AGENTS.md](../AGENTS.md) for the overall execution architecture.

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
