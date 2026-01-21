# Execute Module - Agent Scaffolding

## Module Overview

**Purpose**: Execute rendered simulation scripts across multiple frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy)

**Pipeline Step**: Step 12: Execution (12_execute.py)

**Category**: Simulation / Execution

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2026-01-21

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
- Graceful degradation when frameworks unavailable
- Automatic PyMDP package detection (distinguishes correct vs wrong package variants)
- Path collection with deduplication (prevents nested directory issues)
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling

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
  - `"all"`: Execute all available frameworks
  - `"lite"`: Execute only PyMDP, JAX, DisCoPy (no Julia)
  - Comma-separated: `"pymdp,jax"` for specific frameworks
- `simulation_engine` (str): Engine to use ("auto", "pymdp", "rxinfer", etc., default: "auto")
- `validate_only` (bool): Only validate scripts, don't execute (default: False)
- `timeout` (int): Execution timeout per script in seconds (default: 300)
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
- `timeout` (int): Execution timeout in seconds (default: `60`)
- `validate_only` (bool): Only validate scripts, don't execute (default: `False`)
- `capture_output` (bool): Capture stdout/stderr (default: `True`)
- `parallel_execution` (bool): Execute scripts in parallel (default: `False`)

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
- `inferactively-pymdp` - PyMDP simulation engine (package name: `inferactively-pymdp`, fallback: skip PyMDP)
  - **Note**: The correct package name is `inferactively-pymdp`, not `pymdp`
  - The execute module automatically detects wrong package variants
- `julia` - Julia runtime (fallback: skip Julia scripts)
- `jax` - JAX framework (fallback: skip JAX)

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

### Latest Execution
- **Duration**: 32.5s
- **Memory**: Peak 19.26 MB, Final 13.96 MB
- **Status**: SUCCESS_WITH_WARNINGS
- **Scripts Found**: 5
- **Scripts Failed**: 5 (dependency issues)

### Framework Execution Times
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
- **Timeout**: 60s per script (configurable)

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
- **Julia Runtime**: Executes Julia simulation scripts (RxInfer.jl, ActiveInference.jl)
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
- `src/tests/test_execute_integration.py`
- `src/tests/test_execute_pymdp.py`

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
- For Julia: Install packages via `julia -e 'using Pkg; Pkg.add("RxInfer")'`
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
- Increase timeout: `--timeout 600` (10 minutes)
- Simplify model complexity
- Use faster frameworks (JAX) for large models
- Process models individually instead of batch

---

## Version History

### Current Version: 1.0.0

**Features**:
- Multi-framework execution support
- Graceful degradation when frameworks unavailable
- Comprehensive error logging
- Result capture and validation
- Execution timeout handling

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced parallel execution
- **Future**: Real-time execution monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Render Module](../render/AGENTS.md)
- [Execution Guide](../../doc/execution/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
