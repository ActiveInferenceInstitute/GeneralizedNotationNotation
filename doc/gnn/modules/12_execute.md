# Step 12: Execute — Script Execution Across Frameworks

## Overview

Discovers and executes rendered scripts from Step 11's output. Supports Python and Julia execution environments, captures stdout/stderr, extracts simulation data (beliefs, actions, free energy), and organizes results per framework.

## Usage

```bash
python src/12_execute.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/12_execute.py` (62 lines) |
| Module | `src/execute/` |
| Processor | `src/execute/processor.py` (785 lines) |
| Module function | `process_execute()` |

## Execution Flow

1. **Discover** scripts in `output/11_render_output/` via `find_executable_scripts()`
2. **Execute** each via `subprocess.run()` with timeout and cwd
3. **Extract** simulation data (beliefs, actions, observations, free energy) from stdout
4. **Save** structured results per framework with execution logs

## Key Functions

| Function | Purpose |
|----------|---------|
| `find_executable_scripts()` | Discover Python/Julia scripts by framework directory |
| `execute_single_script()` | Run a script with subprocess, capture output and timing |
| `extract_simulation_data()` | Parse simulation traces from script output |

## Output

- **Directory**: `output/12_execute_output/`
  - `execution_summary.json` — All execution results
  - `execution_report.md` — Human-readable summary
  - `<model>/<framework>/execution_logs/` — Per-script logs
  - `<model>/<framework>/simulation_data/` — Extracted simulation results

## MCP Tools (execute module — 5 real tools)

Registered by `src/execute/mcp.py` via `register_tools()`:

| Tool | Description |
|------|-------------|
| `process_execute` | Run GNN execution pipeline across all frameworks in a directory |
| `execute_gnn_model` | Execute a single GNN model file and capture results |
| `execute_pymdp_simulation` | Run a PyMDP simulation from a rendered Python script |
| `check_execute_dependencies` | Check availability of execution framework dependencies |
| `get_execute_module_info` | Return execute module version and capabilities |

## Source

- **Script**: [src/12_execute.py](../../src/12_execute.py)
