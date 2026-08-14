# Step 12: Execute

## Architectural Mapping

**Orchestrator**: `src/12_execute.py` (111 lines)
**Implementation Layer**: `src/execute/`

## Module Description

This module is responsible for running GNN models that have been rendered into framework-specific simulation code by Step 11 (`11_render.py`).


| Framework | Language | Subfolder | Script Pattern | Status |
|-----------|----------|-----------|----------------|--------|
| **PyMDP** | Python | `pymdp/` | `*_pymdp.py` | ✅ Full support |
| **RxInfer.jl** | Julia | `rxinfer/` | `*_rxinfer.jl` | ✅ Full support |
| **ActiveInference.jl** | Julia | `activeinference_jl/` | `*_activeinference.jl` | ✅ Full support |
| **JAX** | Python | `jax/` | `*_jax.py` | ✅ Full support |
| **DisCoPy** | Python | `discopy/` | `*_discopy.py` | ✅ Full support |
| **PyTorch** | Python | `pytorch/` | `*_pytorch.py` | ✅ Full support |
| **NumPyro** | Python | `numpyro/` | `*_numpyro.py` | ✅ Full support |
| **bnlearn** | Python | `bnlearn/` | `*_bnlearn.py` | ✅ Full support |

JAX, NumPyro, PyTorch, and DisCoPy are **core** dependencies (`uv sync`). If the environment is incomplete, their scripts are **skipped** (not failed). Julia frameworks require Julia installed.

## Agent Identity & Capabilities

# Execute Module - Agent Scaffolding

## Module Overview

**Purpose**: Execute rendered simulation scripts across multiple frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, bnlearn).

**Pipeline Step**: Step 12: Execution (12_execute.py)

**Category**: Simulation / Execution

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-08-07

---

## Core Functionality

### Primary Responsibilities
1. Execute Python simulation scripts (PyMDP, JAX, DisCoPy)
2. Execute Julia simulation scripts (RxInfer.jl, ActiveInference.jl)
3. Capture simulation results and logs
4. Handle execution errors gracefully
5. Generate execution reports

### Key Capabilities
- Multi-framework execution support
- **Skip vs fail**: JAX, NumPyro, PyTorch, and DisCoPy are **core** dependencies; if the environment is incomplete, scripts are **skipped** (not run) and reported as "skipped" — they do not count as execution failures. Repair with `uv sync`. Julia backends still require a local Julia install.
- **Committed Julia environments** with `JULIA_PROJECT` defaulting (see below)
- Graceful degradation when frameworks unavailable
- Automatic PyMDP package detection (distinguishes correct vs wrong package variants)
- Path collection with deduplication (prevents nested directory issues)
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling
- Distributed execution across a Ray or Dask cluster for parallel script/parameter-sweep dispatch (`src/execute/distributed.py`, `--distributed`, `--execution-workers`, `--backend {ray,dask}`)

---

## Julia Execution Environments

Both Julia backends run against **committed** environments checked into the repository, so a run never depends on an ambient Julia depot or a runtime `Pkg.add`.

| Framework | Environment | Pinned contents |
|---|---|---|
| RxInfer.jl | `src/execute/rxinfer/` | The `GnnRxInferModels` package — RxInfer 5.5.0 plus Distributions, JSON, Plots, StatsBase, PrecompileTools. Precompiles the `pomdp`, `continuous`, `hierarchical`, `factored`, and `learning` models loudly: a precompile failure surfaces rather than being swallowed. |
| ActiveInference.jl | `src/execute/activeinference_jl/` | A deliberately **minimal** environment — ActiveInference 0.1.2, Distributions, JSON, StatsBase, and nothing else. |

### `JULIA_PROJECT` defaulting

`_build_execution_environment` (`src/execute/processor.py`) sets `JULIA_PROJECT` to the committed environment matching the script's framework, using `setdefault` — **an explicitly exported `JULIA_PROJECT` still wins**. This is what lets `using GnnRxInferModels` / `using ActiveInference` resolve without an ambient environment, including under test runners whose temporary depot may not exist.

`setup_environment.jl` activates and instantiates the environment (`Pkg.activate()` + `Pkg.instantiate()`); there is no runtime `Pkg.add`.

### Skip semantics

A backend whose dependency is absent produces a **skipped** result (`skipped: true`) carrying an explicit dependency reason, not a failure. Skips are counted separately from failures in the step summary and are excluded from the failure threshold that determines step success — so the completion line reads, per model, in the shape `N succeeded, M skipped (dependency not installed)`. On a fully-provisioned run the remaining skips are the intentionally-unlocked optional backends (PyTorch, which is not locked while GHSA-rrmf-rvhw-rf47 is unpatched).

### Exit-code contract

Rendered RxInfer scripts end with `return results["validation"]["all_valid"] ? 0 : 1`. The validation block — belief validity, normalisation, action range, and for continuous models VFE finiteness — therefore **drives the process exit code**, so inference that runs to completion but produces invalid posteriors is surfaced as a failed script rather than a silent success. The results payload is still written either way, so a failing run remains diagnosable.

---

## API Reference

### Public Functions

#### `process_execute(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main execution function called by orchestrator (12_execute.py). Executes rendered simulation scripts across multiple frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing rendered scripts (typically output from Step 11)
- `output_dir` (Path): Output directory for execution results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `frameworks` (str): Frameworks to execute ("all", "lite", or comma-separated list, default: "all")
  - `"all"`: PyMDP, JAX, DisCoPy, RxInfer.jl, ActiveInference.jl, PyTorch, NumPyro, bnlearn
  - `"lite"`: the Python-only subset — PyMDP, JAX, DisCoPy, bnlearn (no Julia)
  - Comma-separated: `"pymdp,jax"` for specific frameworks; names outside the valid set are filtered out
- `simulation_engine` (str): Engine to use ("auto", "pymdp", "rxinfer", etc., default: "auto")
- `validate_only` (bool): Only validate scripts, don't execute (default: False)
- `timeout` (int): Execution timeout per script in seconds (default: 3600, CLI: `--timeout`)
- `distributed` (bool): Run scripts and parameter sweeps in parallel across a Ray/Dask cluster (default: False, CLI: `--distributed`)
- `execution_workers` (int): Number of local or distributed workers for rendered script execution (default: 1, CLI: `--execution-workers`)
- `backend` (str): Backend for distributed execution, `"ray"` or `"dask"` (default: `"ray"`, CLI: `--backend {ray,dask}`)
- `parallel` (bool): Execute scripts in parallel (default: False)
- `**kwargs`: Additional framework-specific options

**Returns**: `bool` - True if execution succeeded, False otherwise

**Example**:
```python
from execute import process_execute
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_execute(
    target_dir=Path("output/11_render_output"),
    output_dir=Path("output/12_execute_output"),
    verbose=True,
    frameworks="pymdp,jax",
    timeout=600
)
```

#### `execute_simulation_from_gnn(gnn_file: Path, framework: str, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Execute simulation for specific GNN file and framework.

**Parameters**:
- `gnn_file` (Path): Path to GNN file
- `framework` (str): Framework to use ("pymdp", "rxinfer", "activeinference_jl", "jax", "discopy")
- `output_dir` (Path): Output directory for execution results
- `**kwargs`: Framework-specific execution options

**Returns**: `Dict[str, Any]` - Execution results dictionary with:
- `success` (bool): Whether execution succeeded
- `return_code` (int): Process return code
- `stdout` (str): Standard output
- `stderr` (str): Standard error
- `duration` (float): Execution duration in seconds
- `output_files` (List[Path]): Generated output files

#### `get_execution_health_status() -> Dict[str, Any]`
**Description**: Get health status of execution environment and framework availability.

**Returns**: `Dict[str, Any]` - Health status dictionary with:
- `pymdp_available` (bool): PyMDP availability
- `rxinfer_available` (bool): RxInfer.jl availability
- `activeinference_jl_available` (bool): ActiveInference.jl availability
- `jax_available` (bool): JAX availability
- `discopy_available` (bool): DisCoPy availability
- `julia_available` (bool): Julia installation status
- `python_version` (str): Python version
- `julia_version` (Optional[str]): Julia version if available

#### PyMDP Package Detection Functions
**Module**: `execute.pymdp.package_detector`

**Functions**:
- `detect_pymdp_installation() -> Dict[str, Any]`: Detect which PyMDP package variant is installed
  - Returns detection results including `correct_package`, `wrong_package`, `has_agent`, `has_mdp_solver`
- `is_correct_pymdp_package() -> bool`: Check if correct package (inferactively-pymdp) is installed
- `get_pymdp_installation_instructions() -> str`: Get actionable installation instructions
- `validate_pymdp_for_execution() -> Dict[str, Any]`: Validate PyMDP is ready for execution
  - Returns `ready` status, detection results, and installation instructions

**Usage**:
```python
from execute.pymdp.package_detector import detect_pymdp_installation, is_correct_pymdp_package

detection = detect_pymdp_installation()
if detection.get("wrong_package"):
    print("Wrong PyMDP package installed - install inferactively-pymdp")
elif not detection.get("correct_package"):
    print("PyMDP not installed - install inferactively-pymdp")
```

---

## Configuration

### Configuration Options

#### Simulation Engine Selection
- `simulation_engine` (str): Engine to use for execution (default: `"auto"`)
  - `"auto"`: Automatically select best available engine
  - `"pymdp"`: Use PyMDP for Python simulations
  - `"rxinfer"`: Use RxInfer.jl for Julia simulations
  - `"activeinference_jl"`: Use ActiveInference.jl
  - `"jax"`: Use JAX framework
  - `"discopy"`: Use DisCoPy for categorical diagrams

#### Execution Parameters
- `timeout` (int): Execution timeout in seconds (default: `3600`)
- `validate_only` (bool): Only validate scripts, don't execute (default: `False`)
- `capture_output` (bool): Capture stdout/stderr (default: `True`)
- `parallel_execution` (bool): Execute scripts in parallel (default: `False`)

#### Distributed Execution
- `distributed` (bool, CLI: `--distributed`): Run scripts and model parameter sweeps in parallel across a Ray/Dask cluster (default: `False`)
- `execution_workers` (int, CLI: `--execution-workers`): Number of local or distributed workers for rendered script execution (default: `1`)
- `backend` (str, CLI: `--backend {ray,dask}`): Backend to use for distributed execution (default: `"ray"`); implemented by `src/execute/distributed.py`'s `Dispatcher` class

#### Framework-Specific Configuration
- `julia_path` (str): Path to Julia executable (default: auto-detect)
- `python_env` (str): Python environment to use (default: current environment)
- `jax_device` (str): JAX device to use (default: `"cpu"`, options: `"cpu"`, `"gpu"`)

---

## Dependencies

### Required Dependencies
- `subprocess` - Script execution
- `json` - Result serialization

### Optional Dependencies
- `inferactively-pymdp` - PyMDP simulation engine (package name: `inferactively-pymdp`, recovery: skip PyMDP)
  - **Note**: The correct package name is `inferactively-pymdp`, not `pymdp`
  - The execute module automatically detects wrong package variants
- `julia` - Julia runtime (recovery: skip Julia scripts)
- `jax` - JAX framework (recovery: skip JAX)

---

## Usage Examples

### Basic Usage
```python
from execute import process_execute

success = process_execute(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/12_execute_output"),
    simulation_engine="auto"
)
```

### Distributed Execution
```bash
# Fan out rendered scripts / parameter sweeps across a local or remote Ray cluster
python src/12_execute.py --target-dir output/11_render_output --output-dir output \
  --distributed --execution-workers 4 --backend ray

# Use Dask instead of Ray
python src/12_execute.py --target-dir output/11_render_output --output-dir output \
  --distributed --execution-workers 4 --backend dask
```

---

## Output Specification

### Output Products
- `execution_results.json` - Execution results summary
- `execution_report.md` - Human-readable report
- `execution_logs/*.log` - Per-script execution logs
- `simulation_data/*.json` - Simulation output data

### Output Directory Structure
```
output/12_execute_output/
├── execution_results/
│   ├── execution_results.json
│   ├── execution_report.md
│   └── execution_logs/
│       ├── pymdp_simulation.log
│       ├── rxinfer_simulation.log
│       └── activeinference_simulation.log
└── simulation_data/
    └── results_*.json
```

---

## Performance Characteristics

Per-run duration, memory, and per-script outcomes are recorded in `execution_results.json` and the pipeline execution summary. Read those for current numbers rather than hard-coding a snapshot here.

### Framework Execution Times (indicative)
- **PyMDP**: ~1-5 seconds
- **RxInfer.jl**: ~10-20 seconds (JIT compilation)
- **ActiveInference.jl**: ~10-15 seconds
- **JAX**: ~2-8 seconds (with GPU)
- **DisCoPy**: ~1-3 seconds

---

## Error Handling

### Graceful Degradation
- **PyMDP unavailable**: Log warning, skip PyMDP scripts
- **Julia unavailable**: Log warning, skip Julia scripts
- **JAX unavailable**: Log warning, skip JAX scripts
- **Script errors**: Capture stderr, continue with other scripts
- **Timeout**: 3600s per script (configurable via `--timeout`)

### Error Categories
1. **Dependency Errors**: Framework not installed
2. **Syntax Errors**: Generated code has errors
3. **Runtime Errors**: Simulation crashes
4. **Timeout Errors**: Execution exceeds limit

---

## Integration Points

### Pipeline Integration
- **Input**: Receives rendered simulation scripts from Step 11 (render)
- **Output**: Generates execution results for Step 13 (llm analysis), Step 16 (analysis), and Step 23 (report generation)
- **Dependencies**: Requires rendered code from `11_render.py` output

### Module Dependencies
- **render/**: Consumes rendered simulation scripts
- **llm/**: Provides execution results for LLM analysis
- **analysis/**: Provides execution data for statistical analysis
- **report/**: Provides execution summaries for reports

### External Integration
- **PyMDP**: Executes Python Active Inference simulations
- **Julia Runtime**: Executes Julia simulation scripts (RxInfer.jl, ActiveInference.jl) under the committed environments described in [Julia Execution Environments](#julia-execution-environments). Equivalent by hand: `julia --startup-file=no --project=src/execute/rxinfer <script>`.
- **JAX**: Executes JAX-based simulations
- **DisCoPy**: Executes categorical diagram computations

### Data Flow
```
11_render.py (Code generation)
  ↓
12_execute.py (Script execution)
  ↓
  ├→ 13_llm.py (LLM analysis of results)
  ├→ 16_analysis.py (Statistical analysis)
  ├→ 23_report.py (Execution reports)
  └→ output/12_execute_output/ (Execution results)
```

---

## Testing

### Test Files
- `src/tests/execute/test_execute_overall.py`
- `src/tests/execute/test_execute_pymdp_integration.py`
- `src/tests/execute/test_execute_pymdp_package.py`

### Test Coverage
- **Current**: 79%
- **Target**: 85%+

### Key Test Scenarios
1. Multi-framework execution
2. Error handling and recovery
3. Result capture and validation
4. Timeout handling

---

## MCP Integration

### Tools Registered
- `execute.run_simulation` - Execute simulation script
- `execute.validate_environment` - Validate execution environment
- `execute.get_health_status` - Get framework health status
- `execute.analyze_error` - Analyze execution errors

### Tool Endpoints
```python
@mcp_tool("execute.run_simulation")
def run_simulation_tool(script_path: str, framework: str) -> Dict[str, Any]:
    """Execute simulation script"""
    # Implementation
```

### MCP File Location
- `src/execute/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Julia execution fails
**Symptom**: Julia scripts fail to execute  
**Cause**: Julia not installed or not in PATH  
**Solution**: 
- Install Julia: `brew install julia` (macOS) or download from [julialang.org](https://julialang.org)
- Verify Julia installation: `julia --version`
- Check Julia is in PATH: `which julia`
- Install required Julia packages if needed

#### Issue 2: Framework dependencies missing
**Symptom**: Execution fails with import errors  
**Cause**: Required packages not installed in environment  
**Solution**:
- Install framework dependencies: `uv pip install inferactively-pymdp jax`
- **Note**: The correct PyMDP package name is `inferactively-pymdp`, not `pymdp`
- For Julia: the RxInfer.jl environment is pinned by the committed `Project.toml` + `Manifest.toml` under `src/execute/rxinfer/` (RxInfer 5.5.0). `setup_environment.jl` activates and instantiates it (`Pkg.activate()` + `Pkg.instantiate()`) — there is no runtime `Pkg.add`.
- Check framework-specific requirements in documentation

#### Issue 2a: Wrong PyMDP package installed
**Symptom**: Error message "Wrong pymdp package installed. Found 'pymdp' with MDP/MDPSolver"  
**Cause**: The wrong `pymdp` package (with MDP/MDPSolver) is installed instead of `inferactively-pymdp`  
**Solution**:
- Uninstall wrong package: `uv pip uninstall pymdp`
- Install correct package: `uv pip install inferactively-pymdp`
- Or use setup module: `python src/1_setup.py --install_optional --optional_groups pymdp`
- The execute module automatically detects wrong package variants and provides clear error messages

#### Issue 3: Execution timeout
**Symptom**: Scripts timeout before completion  
**Cause**: Simulation too complex or timeout too short  
**Solution**:
- Increase timeout beyond the default 3600s (1 hour): `--timeout 7200` (2 hours)
- Simplify model complexity
- Use faster frameworks (JAX) for large models
- Process models individually instead of batch

---

## Version History

### Current package version

See [pyproject.toml](../../../pyproject.toml).

**Features**:
- Multi-framework execution support
- Graceful degradation when frameworks unavailable
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling
- Distributed execution across a Ray/Dask cluster (`--distributed`, `--execution-workers`, `--backend`)

**Known Issues**:
- None currently

### Roadmap
- **Future**: Real-time execution monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../src/execute/../../README.md)
- [Architecture Guide](../../../src/execute/../../ARCHITECTURE.md)
- [Render Module](../../../src/execute/../render/AGENTS.md)
- [Execution Guide](../../../src/execute/../../doc/execution/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2026-08-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/execute/README.md)**: Module Overview
- **[AGENTS](../../../src/execute/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/execute/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/execute/SKILL.md)**: Capability API


---

**Source Reference**: [src/execute](../../../src/execute)
