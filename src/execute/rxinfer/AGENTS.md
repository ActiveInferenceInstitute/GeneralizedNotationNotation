# RxInfer.jl Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution and simulation of RxInfer.jl (Julia) probabilistic models generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / RxInfer.jl Simulation

---

## Core Functionality

### Primary Responsibilities
1. Execute RxInfer.jl simulations from generated Julia code
2. Run reactive message-passing inference on probabilistic models
3. Manage Julia environment and RxInfer.jl execution
4. Provide analysis and visualization of inference results
5. Handle RxInfer.jl-specific execution parameters and configurations

### Key Capabilities
- Reactive message-passing inference execution
- Julia environment management and package loading
- Real-time inference monitoring and result streaming
- Comprehensive result analysis and visualization
- Error handling and recovery for Julia/RxInfer.jl execution
- Cross-platform compatibility (Linux/macOS/Windows)

---

## API Reference

### Public Functions

#### `execute_rxinfer_simulation(julia_script_path: str, config: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Execute RxInfer.jl simulation from generated Julia script

**Parameters**:
- `julia_script_path` (str): Path to RxInfer.jl simulation script
- `config` (Dict): Execution configuration parameters

**Returns**: Dictionary with simulation results and metadata

**Example**:
```python
from execute.rxinfer import execute_rxinfer_simulation

config = {
    'iterations': 100,
    'convergence_threshold': 1e-6,
    'data_file': 'observations.csv',
    'constraints': 'default',
    'meta': True,
    'visualization': True
}

results = execute_rxinfer_simulation("simulation.jl", config)
print(f"Inference completed in {results['execution_time']:.2f}s")
```

#### `run_rxinfer_inference(model_code: str, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Run RxInfer.jl inference with model code and data

**Parameters**:
- `model_code` (str): RxInfer.jl model code
- `data` (Dict): Observation data dictionary
- `config` (Dict): Inference configuration

**Returns**: Inference results dictionary

#### `validate_julia_environment() -> Dict[str, bool]`
**Description**: Validate Julia environment and RxInfer.jl installation

**Returns**: Validation results dictionary

#### `setup_rxinfer_execution(config: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Setup RxInfer.jl execution environment

**Parameters**:
- `config` (Dict): Setup configuration

**Returns**: Setup results and environment info

---

## Dependencies

### Required Dependencies
- `julia` - Julia programming language runtime
- `RxInfer.jl` - RxInfer.jl Julia package
- `ReactiveMP.jl` - Reactive message passing library
- `GraphPPL.jl` - Probabilistic programming DSL

### Optional Dependencies
- `Plots.jl` - Visualization support (fallback: data export)
- `DataFrames.jl` - Data manipulation (fallback: basic arrays)
- `TOML.jl` - Configuration file support (fallback: JSON)

### Internal Dependencies
- `execute.executor` - Base execution functionality
- `render.rxinfer` - RxInfer.jl code generation
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### RxInfer.jl Execution Configuration
```python
RXINFER_EXEC_CONFIG = {
    'inference': {
        'iterations': 100,              # Maximum inference iterations
        'tolerance': 1e-6,              # Convergence tolerance
        'scheduler': 'Asynchronous',    # Inference scheduler
        'autoupdates': True,            # Automatic model updates
        'meta': True,                   # Use meta-programming
        'constraints': 'default'        # Constraint specification
    },
    'data': {
        'format': 'csv',                # Data format
        'batch_size': 100,              # Batch size for streaming
        'validation_split': 0.2,        # Train/validation split
        'normalization': True           # Data normalization
    },
    'execution': {
        'julia_threads': 4,             # Julia threads
        'memory_limit': '4GB',          # Memory limit
        'timeout': 300,                 # Execution timeout (seconds)
        'cleanup_temp': True            # Clean temporary files
    },
    'output': {
        'save_results': True,           # Save inference results
        'export_format': 'json',        # Export format
        'visualization': True,          # Generate visualizations
        'performance_metrics': True     # Performance monitoring
    }
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    'structure': 'factor_graph',        # Model structure type
    'variables': 'random',             # Variable types
    'factors': 'constraint_based',     # Factor specification
    'priors': 'informative',           # Prior distributions
    'likelihoods': 'gaussian'          # Likelihood functions
}
```

### Performance Configuration
```python
PERFORMANCE_CONFIG = {
    'optimization': {
        'compilation': True,            # JIT compilation
        'caching': True,                # Result caching
        'parallelization': True,        # Parallel execution
        'memory_pool': True             # Memory pooling
    },
    'monitoring': {
        'memory_usage': True,           # Memory monitoring
        'cpu_usage': True,              # CPU monitoring
        'inference_progress': True,     # Progress tracking
        'error_logging': True           # Error logging
    }
}
```

---

## Usage Examples

### Basic RxInfer.jl Simulation Execution
```python
from execute.rxinfer import execute_rxinfer_simulation

# Execute RxInfer.jl simulation
execution_config = {
    'iterations': 50,
    'tolerance': 1e-6,
    'data_file': 'observations.csv',
    'constraints': 'default',
    'meta': True,
    'scheduler': 'Asynchronous',
    'visualization': True
}

results = execute_rxinfer_simulation(
    julia_script_path="output/11_render_output/model_rxinfer_simulation.jl",
    config=execution_config
)

print(f"Inference converged: {results['converged']}")
print(f"Final free energy: {results['free_energy']:.4f}")
print(f"Execution time: {results['execution_time']:.2f}s")
```

### Inference with Custom Data
```python
from execute.rxinfer import run_rxinfer_inference

# Load model code
with open("model.jl", "r") as f:
    model_code = f.read()

# Prepare data
data = {
    'observations': observation_array,
    'controls': control_array,
    'time_steps': 100
}

# Configure inference
inference_config = {
    'iterations': 100,
    'constraints': 'custom',
    'meta': True,
    'scheduler': 'Asynchronous'
}

inference_results = run_rxinfer_inference(model_code, data, inference_config)

# Analyze results
analyze_inference_results(inference_results)
```

### Environment Validation
```python
from execute.rxinfer import validate_julia_environment

# Validate Julia and RxInfer.jl setup
validation = validate_julia_environment()

print("Julia Environment Validation:")
print(f"Julia installed: {validation['julia_available']}")
print(f"Version: {validation.get('julia_version', 'Unknown')}")
print(f"RxInfer.jl available: {validation['rxinfer_available']}")
print(f"ReactiveMP.jl available: {validation['reactivemp_available']}")

if not validation['environment_ready']:
    print("Issues found:")
    for issue in validation['issues']:
        print(f"  - {issue}")
```

### Setup and Execution
```python
from execute.rxinfer import setup_rxinfer_execution

# Setup execution environment
setup_config = {
    'julia_threads': 4,
    'memory_limit': '4GB',
    'working_directory': './rxinfer_work',
    'cleanup_on_exit': True
}

setup_results = setup_rxinfer_execution(setup_config)

if setup_results['setup_successful']:
    print("RxInfer.jl environment ready")
    # Proceed with execution
    execute_simulation()
else:
    print("Setup failed:")
    for error in setup_results['errors']:
        print(f"  - {error}")
```

---

## Reactive Inference Implementation

### Message Passing Algorithms
- **Belief Propagation**: Sum-product algorithm for exact inference
- **Variational Message Passing**: Approximate inference with variational distributions
- **Expectation Propagation**: Moment matching for exponential families
- **Loopy Belief Propagation**: Iterative message passing on cyclic graphs

### Reactive Execution Model
- **Streaming Inference**: Real-time inference on data streams
- **Online Learning**: Incremental model updates with new data
- **Adaptive Scheduling**: Dynamic message scheduling based on convergence
- **Constraint Propagation**: Declarative constraint satisfaction

### Model Types Supported
- **Static Models**: Fixed structure probabilistic models
- **Dynamic Models**: Time-series and sequential models
- **Hierarchical Models**: Multi-level probabilistic hierarchies
- **Hybrid Models**: Combination of discrete and continuous variables

---

## Output Specification

### Output Products
- `rxinfer_results.json` - Complete inference results
- `belief_trajectories.csv` - Belief evolution data
- `free_energy_plot.png` - Free energy minimization plot
- `factor_graph.dot` - Graphviz factor graph representation
- `performance_metrics.json` - Execution performance data

### Output Directory Structure
```
output/12_execute_output/
├── rxinfer_results.json
├── belief_trajectories.csv
├── free_energy_plot.png
├── factor_graph.dot
└── execution_logs/
    └── inference_log.txt
```

### Result Data Structure
```python
inference_results = {
    'metadata': {
        'model_name': 'actinf_pomdp_agent',
        'framework': 'rxinfer',
        'julia_version': '1.8.5',
        'rxinfer_version': '2.4.1',
        'execution_time': 45.67,
        'timestamp': '2025-10-28T10:30:00Z'
    },
    'inference': {
        'converged': True,
        'iterations_used': 87,
        'final_free_energy': -1234.56,
        'convergence_history': [...]
    },
    'posteriors': {
        'variable_1': {
            'mean': [...],
            'variance': [...],
            'samples': [...]
        },
        'variable_2': {...}
    },
    'performance': {
        'memory_peak': '2.3GB',
        'cpu_time': 42.15,
        'messages_passed': 15420,
        'constraints_satisfied': 98.7
    }
}
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 5-120 seconds per inference (depends on model complexity)
- **Memory**: 200-1000MB depending on model size
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Julia Startup**: 2-5s
- **Model Compilation**: 3-10s
- **Inference Execution**: 1-100s (main computation)
- **Result Processing**: 1-5s
- **Visualization**: 2-10s

### Optimization Notes
- RxInfer.jl is highly optimized for message passing
- Julia's JIT compilation improves performance on repeated runs
- Memory usage scales with graph size and variable dimensions
- Parallel execution available for multi-core systems

---

## Error Handling

### RxInfer.jl Execution Errors
1. **Julia Environment Issues**: Julia not found or RxInfer.jl not installed
2. **Model Compilation Errors**: Invalid Julia/RxInfer.jl syntax
3. **Inference Convergence Issues**: Poor model specification
4. **Memory/Resource Limitations**: Insufficient system resources

### Recovery Strategies
- **Environment Validation**: Comprehensive pre-execution checks
- **Model Validation**: Syntax and semantic validation of generated code
- **Parameter Tuning**: Automatic parameter adjustment for convergence
- **Resource Management**: Memory and CPU usage optimization

### Error Examples
```python
try:
    results = execute_rxinfer_simulation(script_path, config)
except RxInferExecutionError as e:
    logger.error(f"RxInfer.jl execution failed: {e}")
    # Attempt recovery with simplified model
    simplified_config = simplify_rxinfer_config(config)
    results = execute_rxinfer_simulation(script_path, simplified_config)
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `render.rxinfer` - RxInfer.jl code generation
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `execute.processor` - Main execution integration
- `tests.test_execute_rxinfer*` - RxInfer.jl execution tests

### Data Flow
```
RxInfer.jl Code Generation → Julia Environment Setup → Model Compilation → Inference Execution → Result Analysis → Visualization
```

---

## Testing

### Test Files
- `src/tests/test_execute_rxinfer_integration.py` - Integration tests
- `src/tests/test_execute_rxinfer_inference.py` - Inference tests
- `src/tests/test_execute_rxinfer_performance.py` - Performance tests

### Test Coverage
- **Current**: 75%
- **Target**: 85%+

### Key Test Scenarios
1. Julia environment validation and setup
2. RxInfer.jl model compilation and execution
3. Inference convergence and accuracy testing
4. Result analysis and visualization
5. Error handling and recovery testing

### Test Commands
```bash
# Run RxInfer.jl execution tests
pytest src/tests/test_execute_rxinfer*.py -v

# Run with coverage
pytest src/tests/test_execute_rxinfer*.py --cov=src/execute/rxinfer --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `execute.run_rxinfer_simulation` - Execute RxInfer.jl simulation
- `execute.validate_julia_env` - Validate Julia environment
- `execute.analyze_rxinfer_results` - Analyze inference results
- `execute.visualize_rxinfer_beliefs` - Visualize belief trajectories

### Tool Endpoints
```python
@mcp_tool("execute.run_rxinfer_simulation")
def run_rxinfer_simulation_tool(script_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute RxInfer.jl simulation with given configuration"""
    return execute_rxinfer_simulation(script_path, config)
```

---

## Reactive Programming Features

### Streaming Inference
- **Real-time Processing**: Continuous inference on data streams
- **Online Learning**: Incremental parameter updates
- **Adaptive Computation**: Dynamic graph restructuring
- **Memory Efficiency**: Streaming algorithms for large datasets

### Constraint Programming
- **Declarative Constraints**: High-level constraint specification
- **Automatic Satisfaction**: Constraint propagation algorithms
- **Custom Constraints**: User-defined constraint functions
- **Constraint Optimization**: Efficient constraint satisfaction

### Monitoring and Debugging
- **Inference Progress**: Real-time convergence monitoring
- **Message Tracking**: Message-passing flow visualization
- **Performance Profiling**: Computational bottleneck identification
- **Error Diagnostics**: Detailed error reporting and suggestions

---

## Development Guidelines

### Adding New RxInfer.jl Features
1. Update execution logic in `rxinfer_runner.py`
2. Add new Julia code templates and wrappers
3. Update environment validation and setup
4. Add comprehensive tests

### Performance Optimization
- Profile Julia code execution bottlenecks
- Optimize message-passing schedules
- Use efficient data structures for large graphs
- Implement parallel inference strategies

---

## Troubleshooting

### Common Issues

#### Issue 1: "Julia not found in PATH"
**Symptom**: Execution fails with Julia command not found
**Cause**: Julia not installed or not in system PATH
**Solution**: Install Julia and ensure it's in PATH, or specify full path

#### Issue 2: "RxInfer.jl package not available"
**Symptom**: Import errors for RxInfer.jl packages
**Cause**: RxInfer.jl or dependencies not installed in Julia
**Solution**: Install required Julia packages using Pkg.add()

#### Issue 3: "Inference not converging"
**Symptom**: Inference fails to converge within iteration limit
**Cause**: Poor model specification or inappropriate parameters
**Solution**: Adjust inference parameters or improve model structure

### Debug Mode
```python
# Enable debug output for RxInfer.jl execution
results = execute_rxinfer_simulation(script_path, config, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete RxInfer.jl simulation execution pipeline
- Reactive message-passing inference implementation
- Julia environment management and validation
- Comprehensive result analysis and visualization
- Real-time inference monitoring
- Extensive error handling and recovery
- MCP tool integration

**Known Limitations**:
- Requires Julia runtime environment
- Large models may require significant memory
- Some advanced RxInfer.jl features need manual configuration

### Roadmap
- **Next Version**: Enhanced streaming inference support
- **Future**: GPU acceleration for inference
- **Advanced**: Integration with RxInfer.jl's latest reactive features

---

## References

### Related Documentation
- [Execute Module](../../execute/AGENTS.md) - Parent execute module
- [RxInfer.jl Render](../render/rxinfer/AGENTS.md) - RxInfer.jl code generation
- [RxInfer.jl Documentation](https://rxinfer.ml/) - Official RxInfer.jl docs

### External Resources
- [Julia Language](https://julialang.org/)
- [Reactive Programming](https://en.wikipedia.org/wiki/Reactive_programming)
- [Message Passing](https://en.wikipedia.org/wiki/Belief_propagation)

---

**Last Updated**: October 28, 2025
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready
