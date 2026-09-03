---
name: gnn-simulation-execution
description: GNN simulation script execution with result capture. Use when running generated simulation scripts, managing execution environments, handling framework dependencies, or capturing simulation outputs and metrics.
---

# GNN Simulation Execution (Step 12)

## Purpose

Executes rendered simulation scripts across all supported frameworks with robust error handling, circuit breaker patterns, retry logic, and comprehensive result capture.

## Key Commands

```bash
# Execute all frameworks
python src/12_execute.py --target-dir input/gnn_files --output-dir output --verbose

# Specific frameworks only
python src/12_execute.py --frameworks "pymdp,jax" --verbose

# Lite preset (PyMDP, JAX, DisCoPy, bnlearn; bnlearn scripts skip at pre-flight)
python src/12_execute.py --frameworks "lite" --verbose
```

## API

```python
from execute import (
    GNNExecutor,
    execute_gnn_model,
    run_simulation,
    execute_pymdp_simulation_from_gnn,
    execute_pymdp_simulation,
    validate_execution_environment,
    process_execute,
    execute_script_safely,
)

# Process execution step (used by pipeline)
process_execute(target_dir, output_dir, verbose=True)

# Execute a rendered model script directly
result = execute_gnn_model("path/to/rendered_script.py", execution_type="pymdp")

# Run a simulation configuration
result = run_simulation(config)

# Validate execution environment before running
env_report = validate_execution_environment()

# Use the GNNExecutor class
engine = GNNExecutor()
```

## Key Exports

- `process_execute` — main pipeline processing function
- `GNNExecutor` — execution engine class
- `execute_gnn_model` / `run_simulation` — model execution functions
- `execute_pymdp_simulation_from_gnn` / `execute_pymdp_simulation` — PyMDP-specific
- `validate_execution_environment` — pre-execution validation
- `execute_script_safely` — safe script execution with error handling

## Execution Flow

The execute processor follows this pipeline:

1. **Script Discovery** — scans `output/11_render_output/` for `.py` and `.jl` files
2. **Framework Filtering** — filters by `--frameworks` parameter (presets or explicit list)
3. **Subprocess Execution** — runs each script with timeout protection and error capture
4. **Result Collection** — gathers stdout, stderr, exit codes, and timing data
5. **Report Generation** — produces per-framework execution summary and comparison

## Framework Presets

| Preset | Frameworks | Use Case |
| ------ | ---------- | -------- |
| `all` | PyMDP, RxInfer, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan | Full execution (default) |
| `lite` | PyMDP, JAX, DisCoPy, bnlearn | Python-only, no Julia required |
| `pymdp,jax` | PyMDP, JAX | Fast Python subset |

## Dependencies

```bash
# Core execution (PyMDP)
uv sync

# For Julia frameworks: instantiate the committed environments (no runtime Pkg.add)
julia --startup-file=no --project=src/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'

# For Stan: cmdstanpy plus a CmdStan toolchain
uv sync --extra stan

# For DisCoPy
uv sync --extra graphs
```

## Output

- Execution results in `output/12_execute_output/`
- Per-framework subdirectories with captured outputs
- Execution summary report (JSON)
- Performance metrics and timing data


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `check_execute_dependencies`
- `execute_gnn_model`
- `execute_pymdp_simulation`
- `get_execute_module_info`
- `process_execute`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
