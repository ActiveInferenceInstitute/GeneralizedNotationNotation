# ActiveInference.jl Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution of ActiveInference.jl simulations generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / Active Inference Simulation

---

## Core Functionality

### Primary Responsibilities
1. Discover rendered ActiveInference.jl scripts under `output/11_render_output/<model>/activeinference_jl/`
2. Set up and validate the committed Julia project (`Project.toml` + `Manifest.toml` in this directory)
3. Execute each script as a Julia subprocess and write a Python-side execution report

### Key Capabilities
- Reproducible Julia environment: `setup_environment.jl` activates and instantiates the committed
  project (Julia 1.12 works; `Distributions` is pinned to `"0.25.100 - 0.25.125"` so `DistributionsAD`
  0.6.58 precompiles)
- Environment status probe (`get_environment_status`) that drives automatic setup at Step 12
- Per-script execution with captured stdout/stderr and a JSON execution report
- Supporting Julia analysis suites shipped alongside the runner
  (`adaptive_precision_attention.jl`, `counterfactual_reasoning.jl`,
  `export_enhancement.jl`, `integration_suite.jl`)

---

## API Reference

Public names re-exported from `execute.activeinference_jl` (`__all__` in `__init__.py`):
`run_activeinference_analysis`, `find_activeinference_scripts`,
`execute_activeinference_script`, `is_julia_available`. All are defined in
`activeinference_runner.py`, which also provides the module-level helpers
`setup_julia_environment` and `get_environment_status`.

### Public Functions

#### `run_activeinference_analysis(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False, force_setup: bool = False) -> bool`
**Description**: Find every script below `<rendered_simulators_dir>/activeinference_jl`, run
`setup_julia_environment` once, execute each script and write
`activeinference_execution_report.json` into the output directory.

**Parameters**:
- `rendered_simulators_dir`: Directory containing rendered simulators (Step 11 output)
- `execution_output_dir`: Where execution outputs go (defaults to `<rendered_simulators_dir>/execution_results/activeinference_jl`)
- `recursive_search`: Whether to walk subdirectories
- `verbose`: Enable verbose logging
- `force_setup`: Re-run the Julia environment setup even if it reports ready

**Returns**: `True` if every script succeeded (or none were found), `False` otherwise.

**Example**:
```python
from pathlib import Path
from execute.activeinference_jl import run_activeinference_analysis

success = run_activeinference_analysis(
    rendered_simulators_dir=Path("output/11_render_output"),
    execution_output_dir=Path("output/12_execute_output"),
    recursive_search=True,
    verbose=True,
)
```

#### `execute_activeinference_script(script_path: Path, verbose: bool = False, output_dir: Optional[Path] = None, setup_environment: bool = True) -> bool`
**Description**: Execute one rendered `.jl` script under the committed project. When
`setup_environment` is `True`, `get_environment_status` is consulted first and
`setup_julia_environment` runs if the environment needs it.

**Returns**: `True` on exit code 0, `False` on failure or a missing script.

#### `find_activeinference_scripts(search_dir: Union[str, Path], recursive: bool = True, include_patterns: Optional[List[str]] = None) -> List[Path]`
**Description**: Collect ActiveInference.jl scripts under `search_dir`.

#### `is_julia_available(min_version: tuple = (1, 9, 0)) -> bool`
**Description**: Return `True` when a `julia` executable is on `PATH` (shared helper from `execute.julia_setup`).

### Module Helpers

#### `setup_julia_environment(project_dir: Path, force_setup: bool = False, verbose: bool = False) -> bool`
Runs `setup_environment.jl` for the given project directory.

#### `get_environment_status(project_dir: Path) -> Dict[str, Any]`
Returns `julia_available`, `project_toml_exists`, `manifest_toml_exists`,
`setup_script_exists`, `environment_report_exists`, `core_packages_status`
(ActiveInference, Distributions, LinearAlgebra, Random, Statistics) and a
`setup_recommendation` of `install_julia`, `create_project`, `run_setup` or `ready`.

---

## Dependencies

### Required Dependencies
- `julia` runtime (Julia 1.12 works with the committed pins)
- `ActiveInference.jl`, `Distributions`, `JSON`, `StatsBase` — pinned by `Project.toml` + `Manifest.toml` in this directory

### Internal Dependencies
- `execute.julia_setup` - Shared Julia availability check
- `execute.processor` - Step 12 orchestration
- `render.activeinference_jl` - Produces the scripts this module executes

---

## Configuration

This module carries no configuration dictionaries of its own. Timeouts and the
framework selection come from Step 12 (`--frameworks`, `timeout` kwarg in
`execute.processor`); the Julia environment is fixed by the committed
`Project.toml` + `Manifest.toml`.

---

## Usage Examples

### Execute one rendered script
```python
from pathlib import Path
from execute.activeinference_jl import execute_activeinference_script, find_activeinference_scripts

for script in find_activeinference_scripts(Path("output/11_render_output"), recursive=True):
    ok = execute_activeinference_script(script, verbose=True, setup_environment=True)
```

### Environment status
```python
from pathlib import Path
from execute.activeinference_jl.activeinference_runner import (
    get_environment_status,
    setup_julia_environment,
)

project_dir = Path("src/execute/activeinference_jl")
status = get_environment_status(project_dir)
print(status["setup_recommendation"])
if status["setup_recommendation"] == "run_setup":
    setup_julia_environment(project_dir, verbose=True)
```

---

## Output Specification

### Output Products
- `simulation_results.json` - Serialized Julia inference results (schema `activeinference_jl_simulation_v1`, or `activeinference_jl_stigmergic_swarm_v1` for the native multi-agent path)
- `activeinference_execution_report.json` - Python-side execution report (written by `run_activeinference_analysis`)
- Execution logs (stdout/stderr) captured by Step 12

### Output Directory Structure
```
output/12_execute_output/
├── <model>/activeinference_jl/
│   └── simulation_results.json
├── activeinference_execution_report.json
└── summaries/
    ├── execution_summary.json
    └── execution_report.md
```

Read result fields from a generated `simulation_results.json`; the renderer in
`src/render/activeinference_jl/activeinference_renderer.py` is the source of truth
for the schema.

---

## Performance Characteristics

Do not treat timing numbers in documentation as current measurements. The first
run pays Julia precompilation for the committed environment; later runs reuse the
compiled cache. Read durations from `summaries/execution_summary.json`.

---

## Error Handling

### Julia/ActiveInference.jl Errors
1. **Julia Not Found**: `is_julia_available()` is `False`; Step 12 reports the framework as skipped
2. **Environment Not Ready**: `get_environment_status` recommends `run_setup`; `setup_julia_environment` is invoked automatically
3. **Script Failure**: the Julia process exits non-zero; `execute_activeinference_script` returns `False`

### Recovery Strategies
- Run `julia --project=src/execute/activeinference_jl --startup-file=no src/execute/activeinference_jl/setup_environment.jl`
- Pass `force_setup=True` to `run_activeinference_analysis` to rebuild the environment
- There is no automatic retry or reduced-analysis fallback; read the captured stderr

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `execute.julia_setup` - Julia availability helper

### Imported By
- `execute.processor` - Main execution integration

### Data Flow
```
render.activeinference_jl (.jl script) → setup_environment.jl → julia --project=src/execute/activeinference_jl → simulation_results.json → Step 12 summaries → Step 16 analysis
```

---

## Testing

### Test Files
- `src/tests/execute/test_execute_overall.py` - Execute module tests (includes framework selection)
- `src/tests/pipeline/test_pomdp_gridworld_cross_framework.py` - Cross-framework contract including ActiveInference.jl
- `src/tests/render/test_activeinference_matrix_formatting.py` - Renderer-side contract for the scripts this module executes

### Test Commands
```bash
uv run --extra dev python -m pytest src/tests/execute/test_execute_overall.py \
    src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -v

# With coverage
uv run --extra dev python -m pytest src/tests/execute/test_execute_overall.py \
    --cov=src/execute/activeinference_jl --cov-report=term-missing
```

---

## MCP Integration

This submodule registers no MCP tools of its own. The parent module
(`src/execute/mcp.py`) exposes `process_execute`, `execute_gnn_model`,
`execute_pymdp_simulation`, `check_execute_dependencies` and
`get_execute_module_info`; ActiveInference.jl scripts are reached through
`process_execute` with the `activeinference_jl` framework selected.

---

## Development Guidelines

### Adding New Features
1. Change the generated Julia in `src/render/activeinference_jl/` (this module only runs what the renderer emits)
2. Extend the Julia analysis suites here (`integration_suite.jl` and siblings) when needed
3. Update Python wrapper functions in `activeinference_runner.py`
4. Update `Project.toml`/`Manifest.toml` together and re-run `setup_environment.jl`
5. Add or extend tests in `src/tests/execute/` and `src/tests/render/`

---

## Troubleshooting

### Common Issues

#### Issue 1: "Julia command not found"
**Symptom**: Step 12 reports the ActiveInference.jl framework as skipped
**Cause**: Julia not installed or not in system PATH
**Solution**: Install Julia and ensure `julia` resolves on PATH

#### Issue 2: "ActiveInference.jl package not available"
**Symptom**: `using ActiveInference` fails inside the script
**Cause**: Committed environment not instantiated
**Solution**: Run `setup_environment.jl` (see Recovery Strategies) or pass `force_setup=True`

#### Issue 3: Precompilation failure on Julia 1.12
**Symptom**: `DistributionsAD` fails to precompile
**Cause**: `Distributions >= 0.25.126` changed `@check_args`
**Solution**: Keep the committed pin `Distributions = "0.25.100 - 0.25.125"`; `setup_environment.jl` also patches the ReverseDiff extension before precompilation

---

## Version History

### Current Version: 3.2.0

**Features**:
- ActiveInference.jl script discovery and subprocess execution under the committed project
- Automatic environment setup driven by `get_environment_status`
- Python-side execution report

**Known Limitations**:
- Requires a local Julia runtime
- Continuous-state models are reported as `unsupported` at render time and never reach this module

---

## References

### Related Documentation
- [Execute Module](../AGENTS.md) - Parent execute module
- [ActiveInference.jl Render](../../render/activeinference_jl/AGENTS.md) - ActiveInference.jl code generation
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) - Active Inference theory

### External Resources
- [Julia Language](https://julialang.org/)
- [ActiveInference.jl](https://github.com/ilabcode/ActiveInference.jl)
- [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)

---

**Last Updated**: 2026-09-02
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready
