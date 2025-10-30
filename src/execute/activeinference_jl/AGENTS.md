# ActiveInference.jl Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution and analysis of ActiveInference.jl simulations generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / Active Inference Analysis

---

## Core Functionality

### Primary Responsibilities
1. Execute ActiveInference.jl simulation scripts generated from GNN models
2. Perform comprehensive analysis of Active Inference simulations
3. Manage Julia environment and ActiveInference.jl package execution
4. Generate detailed analysis reports and visualizations
5. Handle multi-level Active Inference model execution and validation

### Key Capabilities
- Complete ActiveInference.jl simulation execution pipeline
- Hierarchical Active Inference model analysis
- Temporal dynamics and planning evaluation
- Meta-cognitive analysis and uncertainty quantification
- Adaptive precision and attention mechanism analysis
- Counterfactual reasoning and multi-scale temporal analysis
- Statistical analysis and performance metrics computation

---

## API Reference

### Public Functions

#### `run_activeinference_analysis(pipeline_output_dir: str, analysis_type: str = "comprehensive", **kwargs) -> bool`
**Description**: Main function to run ActiveInference.jl analysis on simulation outputs

**Parameters**:
- `pipeline_output_dir` (str): Directory containing pipeline outputs
- `analysis_type` (str): Type of analysis ("basic", "comprehensive", "all")
- `**kwargs`: Additional options (recursive_search, verbose, output_dir)

**Returns**: `True` if analysis completed successfully

**Example**:
```python
from execute.activeinference_jl import run_activeinference_analysis

success = run_activeinference_analysis(
    pipeline_output_dir="output/11_render_output",
    analysis_type="comprehensive",
    recursive_search=True,
    verbose=True
)
```

#### `execute_activeinference_jl_script(script_path: str, config: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Execute a specific ActiveInference.jl script with given configuration

**Parameters**:
- `script_path` (str): Path to ActiveInference.jl script
- `config` (Dict): Execution configuration parameters

**Returns**: Dictionary with execution results and analysis data

#### `analyze_activeinference_results(results_dir: str, analysis_config: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Perform comprehensive analysis on ActiveInference.jl simulation results

**Parameters**:
- `results_dir` (str): Directory containing simulation results
- `analysis_config` (Dict): Analysis configuration parameters

**Returns**: Dictionary with analysis results and metrics

#### `validate_julia_activeinference_environment() -> Dict[str, bool]`
**Description**: Validate Julia and ActiveInference.jl environment setup

**Parameters**: None

**Returns**: Dictionary with validation results for each component

---

## Dependencies

### Required Dependencies
- `julia` - Julia programming language runtime
- `ActiveInference.jl` - ActiveInference.jl package
- `subprocess` - Python subprocess management
- `pathlib` - Path manipulation utilities

### Optional Dependencies
- `numpy` - Numerical computations (fallback: basic arrays)
- `pandas` - Data analysis (fallback: basic data structures)
- `matplotlib` - Visualization (fallback: no plotting)

### Internal Dependencies
- `execute.executor` - Base execution functionality
- `render.activeinference_jl` - ActiveInference.jl code generation
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Analysis Configuration
```python
ANALYSIS_CONFIG = {
    'analysis_type': 'comprehensive',        # Analysis depth
    'recursive_search': True,                # Search subdirectories
    'verbose': True,                         # Verbose output
    'output_format': 'both',                 # Output format (json, markdown, both)
    'performance_monitoring': True,          # Monitor performance
    'error_handling': 'robust',              # Error handling strategy
    'parallel_execution': False              # Parallel analysis
}
```

### Execution Configuration
```python
EXECUTION_CONFIG = {
    'julia_executable': 'julia',             # Julia executable path
    'working_directory': './activeinference_work',  # Working directory
    'timeout_seconds': 300,                  # Execution timeout
    'memory_limit': '4GB',                   # Memory limit
    'cpu_cores': 4,                          # CPU cores to use
    'environment_variables': {               # Environment variables
        'JULIA_NUM_THREADS': '4',
        'JULIA_PROJECT': '@.'
    }
}
```

### Analysis Types Configuration
```python
ANALYSIS_TYPES = {
    'basic': {
        'components': ['simulation', 'basic_metrics'],
        'depth': 'minimal'
    },
    'comprehensive': {
        'components': ['simulation', 'planning', 'learning', 'analysis'],
        'depth': 'full'
    },
    'all': {
        'components': ['all_available'],
        'depth': 'complete'
    }
}
```

---

## Usage Examples

### Basic ActiveInference.jl Analysis
```python
from execute.activeinference_jl import run_activeinference_analysis

# Run comprehensive analysis on pipeline outputs
success = run_activeinference_analysis(
    pipeline_output_dir="output/11_render_output",
    analysis_type="comprehensive",
    recursive_search=True,
    verbose=True
)

if success:
    print("ActiveInference.jl analysis completed successfully")
else:
    print("Analysis failed - check logs for details")
```

### Execute Specific Script
```python
from execute.activeinference_jl import execute_activeinference_jl_script

# Execute specific ActiveInference.jl script
config = {
    'timeout_seconds': 600,
    'memory_limit': '8GB',
    'analysis_type': 'comprehensive'
}

results = execute_activeinference_jl_script(
    script_path="output/11_render_output/model_activeinference_simulation.jl",
    config=config
)

print(f"Execution time: {results['execution_time']:.2f}s")
print(f"Analysis completed: {results['analysis_successful']}")
```

### Environment Validation
```python
from execute.activeinference_jl import validate_julia_activeinference_environment

# Validate Julia and ActiveInference.jl setup
validation = validate_julia_activeinference_environment()

print("Environment Validation:")
print(f"Julia installed: {validation['julia_available']}")
print(f"ActiveInference.jl available: {validation['activeinference_available']}")
print(f"Required packages: {validation['packages_available']}")
print(f"Environment ready: {validation['environment_ready']}")

if not validation['environment_ready']:
    print("Missing components:")
    for component, status in validation.items():
        if not status and component != 'environment_ready':
            print(f"  - {component}")
```

### Results Analysis
```python
from execute.activeinference_jl import analyze_activeinference_results

# Analyze simulation results
analysis_config = {
    'metrics': ['free_energy', 'planning_accuracy', 'learning_curves'],
    'visualization': True,
    'statistical_tests': True,
    'output_format': 'markdown'
}

analysis_results = analyze_activeinference_results(
    results_dir="output/12_execute_output/activeinference_results",
    analysis_config=analysis_config
)

print(f"Analysis completed for {len(analysis_results['trials'])} trials")
print(f"Average free energy: {analysis_results['summary']['avg_free_energy']:.3f}")
```

---

## Active Inference Analysis Components

### Simulation Execution
- **Model Loading**: Load and validate ActiveInference.jl models
- **Simulation Running**: Execute simulations with proper parameters
- **Result Capture**: Capture simulation outputs and trajectories
- **Performance Monitoring**: Monitor execution time and resource usage

### Comprehensive Analysis Suite
- **Statistical Analysis**: Compute statistical metrics and distributions
- **Uncertainty Quantification**: Analyze uncertainty in beliefs and actions
- **Meta-cognitive Analysis**: Evaluate meta-cognitive performance
- **Adaptive Precision Analysis**: Analyze precision adaptation mechanisms
- **Counterfactual Reasoning**: Evaluate counterfactual reasoning capabilities
- **Multi-scale Temporal Analysis**: Analyze temporal dynamics across scales
- **Advanced POMDP Analysis**: Perform advanced partially observable MDP analysis

### Visualization and Reporting
- **Enhanced Visualization**: Create comprehensive result visualizations
- **Integration Testing**: Test model integration and consistency
- **Export Enhancement**: Enhanced data export and formatting
- **Visualization Utilities**: General-purpose visualization tools

---

## Output Specification

### Output Products
- `activeinference_analysis_report.md` - Comprehensive analysis report
- `activeinference_results.json` - Detailed analysis results
- `activeinference_visualizations/` - Analysis visualizations
- `activeinference_execution_logs.txt` - Execution logs
- `activeinference_performance_metrics.json` - Performance metrics

### Output Directory Structure
```
output/12_execute_output/
├── activeinference_results/
│   ├── activeinference_analysis_report.md
│   ├── activeinference_results.json
│   ├── activeinference_visualizations/
│   │   ├── free_energy_plot.png
│   │   ├── belief_trajectories.png
│   │   └── planning_performance.png
│   ├── activeinference_execution_logs.txt
│   └── activeinference_performance_metrics.json
└── julia_environment_check.json
```

### Analysis Results Structure
```python
analysis_results = {
    'metadata': {
        'model_name': 'actinf_pomdp_agent',
        'framework': 'activeinference_jl',
        'analysis_type': 'comprehensive',
        'execution_time': 245.67,
        'julia_version': '1.8.5',
        'timestamp': '2025-10-28T10:30:00Z'
    },
    'simulation': {
        'trials_completed': 50,
        'avg_execution_time': 4.2,
        'success_rate': 0.98
    },
    'analysis': {
        'free_energy': {
            'mean': -1250.34,
            'std': 45.67,
            'trajectory': [...]
        },
        'planning_accuracy': {
            'mean': 0.87,
            'std': 0.05,
            'by_trial': [...]
        },
        'belief_accuracy': {
            'mean': 0.92,
            'std': 0.03,
            'evolution': [...]
        }
    },
    'visualizations': [
        'free_energy_trajectory.png',
        'belief_evolution.png',
        'action_selection_patterns.png'
    ],
    'performance': {
        'memory_peak': '3.2GB',
        'cpu_time': 198.45,
        'analysis_completion_rate': 0.96
    }
}
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 2-15 minutes per comprehensive analysis
- **Memory**: 500MB-4GB depending on model complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Julia Environment Setup**: < 5s
- **Model Loading**: < 10s
- **Simulation Execution**: 1-10 minutes (main computation)
- **Analysis Processing**: 30s-2 minutes
- **Visualization Generation**: 20s-1 minute

### Optimization Notes
- Julia's JIT compilation improves performance on repeated runs
- Memory usage scales with model size and analysis depth
- Parallel execution available for multiple trials
- GPU acceleration available for certain computations

---

## Error Handling

### Julia/ActiveInference.jl Errors
1. **Julia Not Found**: Julia executable not in PATH
2. **Package Not Installed**: ActiveInference.jl or dependencies missing
3. **Model Loading Errors**: Invalid model files or syntax errors
4. **Simulation Failures**: Runtime errors during simulation execution
5. **Analysis Errors**: Errors during result analysis and processing

### Recovery Strategies
- **Environment Validation**: Comprehensive pre-execution validation
- **Graceful Degradation**: Continue with available analysis components
- **Fallback Analysis**: Use basic analysis when advanced features fail
- **Detailed Logging**: Comprehensive error reporting and diagnostics

### Error Examples
```python
try:
    results = run_activeinference_analysis(pipeline_output_dir, analysis_type="comprehensive")
except ActiveInferenceExecutionError as e:
    logger.error(f"ActiveInference.jl analysis failed: {e}")
    # Attempt recovery with basic analysis
    results = run_activeinference_analysis(pipeline_output_dir, analysis_type="basic")
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `render.activeinference_jl` - ActiveInference.jl code generation
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `execute.processor` - Main execution integration
- `tests.test_execute_activeinference*` - ActiveInference.jl-specific tests

### Data Flow
```
ActiveInference.jl Code Generation → Julia Environment Setup → Model Compilation → Simulation Execution → Comprehensive Analysis → Visualization → Report Generation
```

---

## Testing

### Test Files
- `src/tests/test_execute_activeinference_integration.py` - Integration tests
- `src/tests/test_execute_activeinference_analysis.py` - Analysis tests
- `src/tests/test_execute_activeinference_performance.py` - Performance tests

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

### Key Test Scenarios
1. Julia environment validation and setup
2. ActiveInference.jl model loading and execution
3. Analysis pipeline end-to-end testing
4. Result analysis and visualization accuracy
5. Error handling and recovery testing

### Test Commands
```bash
# Run ActiveInference.jl execution tests
pytest src/tests/test_execute_activeinference*.py -v

# Run with coverage
pytest src/tests/test_execute_activeinference*.py --cov=src/execute/activeinference_jl --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `execute.run_activeinference_analysis` - Run ActiveInference.jl analysis
- `execute.validate_julia_environment` - Validate Julia environment
- `execute.analyze_activeinference_results` - Analyze simulation results
- `execute.visualize_activeinference_data` - Visualize analysis results

### Tool Endpoints
```python
@mcp_tool("execute.run_activeinference_analysis")
def run_activeinference_analysis_tool(pipeline_output_dir: str, analysis_type: str) -> Dict[str, Any]:
    """Run comprehensive ActiveInference.jl analysis on pipeline outputs"""
    return run_activeinference_analysis(pipeline_output_dir, analysis_type)
```

---

## Active Inference Analysis Features

### Hierarchical Analysis
- **Multi-level Planning**: Analyze planning across hierarchical levels
- **Temporal Abstraction**: Evaluate temporal abstraction mechanisms
- **Goal-directed Behavior**: Assess goal-directed action selection
- **Adaptive Precision**: Analyze precision adaptation strategies

### Advanced Analytical Methods
- **Meta-cognitive Metrics**: Evaluate meta-cognitive performance
- **Uncertainty Quantification**: Quantify uncertainty in all components
- **Counterfactual Analysis**: Analyze counterfactual reasoning
- **Multi-scale Dynamics**: Analyze dynamics across temporal scales
- **Statistical Validation**: Comprehensive statistical testing

### Visualization Capabilities
- **Free Energy Landscapes**: Visualize free energy minimization
- **Belief Trajectories**: Show belief evolution over time
- **Planning Performance**: Visualize planning accuracy and efficiency
- **Learning Curves**: Show parameter learning progression
- **Comparative Analysis**: Compare different model configurations

---

## Development Guidelines

### Adding New Analysis Features
1. Update analysis logic in appropriate Julia files
2. Add new analysis functions in `enhanced_analysis_suite.jl`
3. Update Python wrapper functions in `activeinference_runner.py`
4. Add comprehensive tests for new features

### Performance Optimization
- Profile Julia code execution bottlenecks
- Optimize data transfer between Python and Julia
- Use efficient data structures for large result sets
- Implement parallel analysis when possible

---

## Troubleshooting

### Common Issues

#### Issue 1: "Julia command not found"
**Symptom**: Execution fails with Julia not found error
**Cause**: Julia not installed or not in system PATH
**Solution**: Install Julia and ensure it's accessible, or specify full path in config

#### Issue 2: "ActiveInference.jl package not available"
**Symptom**: Import errors for ActiveInference.jl components
**Cause**: Package not installed in Julia environment
**Solution**: Install ActiveInference.jl using Julia's package manager

#### Issue 3: "Memory allocation failed"
**Symptom**: Execution fails with memory errors during large simulations
**Cause**: Insufficient memory for model size or analysis depth
**Solution**: Reduce model complexity, decrease analysis scope, or increase memory limits

### Debug Mode
```python
# Enable debug output for ActiveInference.jl execution
results = run_activeinference_analysis(
    pipeline_output_dir,
    analysis_type="comprehensive",
    debug=True,
    verbose=True
)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete ActiveInference.jl execution and analysis pipeline
- Hierarchical Active Inference model support
- Comprehensive analysis suite with multiple analysis types
- Julia environment validation and management
- Advanced visualization and reporting capabilities
- Extensive error handling and recovery
- MCP tool integration

**Known Limitations**:
- Requires Julia runtime environment with ActiveInference.jl
- Large hierarchical models may require significant memory
- Some advanced analysis features need manual configuration

### Roadmap
- **Next Version**: Enhanced parallel analysis support
- **Future**: GPU acceleration for analysis computations
- **Advanced**: Integration with latest Active Inference research methods

---

## References

### Related Documentation
- [Execute Module](../../execute/AGENTS.md) - Parent execute module
- [ActiveInference.jl Render](../render/activeinference_jl/AGENTS.md) - ActiveInference.jl code generation
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) - Active Inference theory

### External Resources
- [Julia Language](https://julialang.org/)
- [ActiveInference.jl](https://github.com/ilabcode/ActiveInference.jl)
- [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)

---

**Last Updated**: October 28, 2025
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready




