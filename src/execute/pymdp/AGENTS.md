# PyMDP Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution and simulation of PyMDP (Python Markov Decision Process) models generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / PyMDP Simulation

---

## Core Functionality

### Primary Responsibilities
1. Execute PyMDP simulations from generated code
2. Run inference and learning algorithms on PyMDP agents
3. Manage simulation environments and data collection
4. Provide visualization and analysis of PyMDP simulation results
5. Handle PyMDP-specific execution parameters and configurations

### Key Capabilities
- Complete PyMDP simulation execution pipeline
- Active Inference algorithm implementation
- Multi-trial simulation management
- Real-time visualization during execution
- Performance monitoring and result analysis
- Error handling and recovery for simulation failures

---

## API Reference

### Exported Functions from `__init__.py`

#### `execute_pymdp_simulation_from_gnn(gnn_file: Path, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Main function exported from the module. Execute PyMDP simulation from GNN file.

**Parameters**:
- `gnn_file` (Path): Path to GNN specification file
- `output_dir` (Path): Output directory for simulation results
- `**kwargs`: Additional execution options:
  - `num_trials` (int): Number of simulation trials (default: 100)
  - `trial_length` (int): Length of each trial (default: 50)
  - `timeout` (int): Execution timeout in seconds (default: 300)
  - `visualization` (bool): Enable visualization (default: True)

**Returns**: `Dict[str, Any]` - Simulation results dictionary

**Location**: `src/execute/pymdp/executor.py`

**Example**:
```python
from execute.pymdp import execute_pymdp_simulation_from_gnn
from pathlib import Path

results = execute_pymdp_simulation_from_gnn(
    gnn_file=Path("input/model.md"),
    output_dir=Path("output/pymdp_results"),
    num_trials=100,
    trial_length=50
)
```

#### `execute_pymdp_simulation(model_path: Union[str, Path], config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]`
**Description**: Execute a complete PyMDP simulation from model file.

**Parameters**:
- `model_path` (Union[str, Path]): Path to PyMDP model file
- `config` (Dict[str, Any], optional): Simulation configuration parameters (default: {})
- `num_trials` (int, optional): Number of simulation trials (default: 100)
- `trial_length` (int, optional): Length of each trial (default: 50)
- `timeout` (int, optional): Execution timeout in seconds (default: 300)
- `capture_output` (bool, optional): Capture stdout/stderr (default: True)
- `**kwargs`: Additional execution options

**Returns**: `Dict[str, Any]` - Simulation results dictionary with:
- `success` (bool): Whether execution succeeded
- `trials_completed` (int): Number of completed trials
- `results` (List[Dict]): Trial results
- `execution_time` (float): Total execution time
- `output_files` (List[Path]): Generated output files

**Location**: `src/execute/pymdp/executor.py`

#### `validate_pymdp_environment() -> Dict[str, Any]`
**Description**: Validate PyMDP environment and dependencies.

**Returns**: `Dict[str, Any]` - Validation results

**Location**: `src/execute/pymdp/validator.py`

#### `detect_pymdp_installation() -> Dict[str, Any]`
**Description**: Detect PyMDP package installation and version.

**Returns**: `Dict[str, Any]` - Installation information

**Location**: `src/execute/pymdp/package_detector.py`

#### `run_pymdp_inference(agent: Any, observations: Union[List, np.ndarray], config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]`
**Description**: Run PyMDP inference on observation sequence.

**Parameters**:
- `agent` (Any): PyMDP agent instance
- `observations` (Union[List, np.ndarray]): Observation sequence
- `config` (Dict[str, Any], optional): Inference configuration (default: {})
- `iterations` (int, optional): Number of inference iterations (default: 10)
- `threshold` (float, optional): Convergence threshold (default: 1e-4)
- `**kwargs`: Additional inference options

**Returns**: `Dict[str, Any]` - Inference results with beliefs, predictions, free energy

#### `pymdp_simulation_loop(agent: Any, environment: Any, config: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]`
**Description**: Main simulation loop for PyMDP agent-environment interaction.

**Parameters**:
- `agent` (Any): PyMDP agent instance
- `environment` (Any): Environment simulator
- `config` (Dict[str, Any], optional): Simulation configuration (default: {})
- `num_steps` (int, optional): Number of simulation steps (default: 50)
- `**kwargs`: Additional simulation options

**Returns**: `List[Dict[str, Any]]` - List of step results with observations, actions, beliefs

#### `validate_pymdp_simulation_setup(model_path: Union[str, Path]) -> Dict[str, bool]`
**Description**: Validate PyMDP simulation setup before execution.

**Parameters**:
- `model_path` (Union[str, Path]): Path to simulation file

**Returns**: `Dict[str, bool]` - Validation results with:
- `file_exists` (bool): Whether file exists
- `pymdp_available` (bool): Whether PyMDP is installed
- `dependencies_met` (bool): Whether all dependencies are available
- `code_valid` (bool): Whether code syntax is valid

---

## Dependencies

### Required Dependencies
- `pymdp` - PyMDP library for Active Inference
- `numpy` - Numerical computations
- `matplotlib` - Visualization and plotting
- `pandas` - Data manipulation and analysis

### Optional Dependencies
- `plotly` - Interactive visualizations (fallback: matplotlib)
- `seaborn` - Statistical visualization (fallback: matplotlib)
- `tqdm` - Progress bars (fallback: basic progress)

### Internal Dependencies
- `execute.executor` - Base execution functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### PyMDP Execution Configuration
```python
PYMDP_EXEC_CONFIG = {
    'simulation': {
        'num_trials': 100,              # Number of simulation trials
        'trial_length': 50,             # Length of each trial
        'random_seed': 42,              # Random seed for reproducibility
        'parallel_execution': False,    # Parallel trial execution
        'save_intermediate': False      # Save intermediate results
    },
    'inference': {
        'method': 'variational',        # Inference method
        'iterations': 16,               # Maximum inference iterations
        'threshold': 1e-4,              # Convergence threshold
        'damping': 0.9,                 # Message damping factor
        'learning_rate': 0.1            # Learning rate for updates
    },
    'environment': {
        'type': 'grid_world',           # Environment type
        'size': [10, 10],               # Environment dimensions
        'noise_level': 0.1,             # Observation noise
        'reward_structure': 'sparse'    # Reward distribution
    },
    'visualization': {
        'enabled': True,                # Enable visualization
        'update_frequency': 10,         # Update frequency
        'save_frames': False,           # Save animation frames
        'plot_types': ['beliefs', 'actions', 'rewards']  # Plot types
    }
}
```

### Agent Configuration
```python
AGENT_CONFIG = {
    'policy_type': 'stochastic',        # Policy type
    'planning_horizon': 1,             # Planning horizon
    'use_posterior': True,             # Use posterior for action selection
    'action_selection': 'deterministic', # Action selection method
    'gamma': 16.0,                     # Precision parameter
    'alpha': 16.0                      # Action precision
}
```

### Result Analysis Configuration
```python
ANALYSIS_CONFIG = {
    'metrics': ['accuracy', 'free_energy', 'surprise'],
    'statistical_tests': ['t_test', 'anova'],
    'confidence_intervals': True,
    'effect_sizes': True,
    'correlation_analysis': True,
    'time_series_analysis': True
}
```

---

## Usage Examples

### Basic PyMDP Simulation Execution
```python
from execute.pymdp import execute_pymdp_simulation

# Execute simulation from generated PyMDP code
simulation_config = {
    'num_trials': 50,
    'trial_length': 20,
    'inference_params': {
        'method': 'variational',
        'iterations': 16,
        'threshold': 1e-4
    },
    'visualization': True,
    'environment': {
        'type': 'grid_world',
        'size': [5, 5]
    }
}

results = execute_pymdp_simulation(
    model_path="output/11_render_output/model_pymdp_simulation.py",
    config=simulation_config
)

print(f"Completed {results['num_trials']} trials")
print(f"Average free energy: {results['avg_free_energy']:.3f}")
```

### Inference Execution
```python
from execute.pymdp import run_pymdp_inference
import pymdp

# Load PyMDP model
model = load_pymdp_model("model.py")
agent = pymdp.Agent(A=model['A'], B=model['B'], C=model['C'], D=model['D'])

# Run inference on observation sequence
observations = generate_observation_sequence(length=100)
inference_config = {
    'iterations': 16,
    'threshold': 1e-4,
    'track_beliefs': True
}

inference_results = run_pymdp_inference(agent, observations, inference_config)

# Analyze inference results
plot_belief_evolution(inference_results['beliefs'])
```

### Complete Simulation Pipeline
```python
from execute.pymdp import pymdp_simulation_loop, create_pymdp_environment

# Setup agent and environment
agent = create_pymdp_agent(model_data)
environment = create_pymdp_environment(env_config)

# Run simulation loop
simulation_config = {
    'num_trials': 10,
    'trial_length': 50,
    'inference_params': {'iterations': 10},
    'learning_enabled': True,
    'visualization': True
}

trial_results = pymdp_simulation_loop(agent, environment, simulation_config)

# Analyze results
analyze_simulation_results(trial_results)
create_performance_plots(trial_results)
```

### Validation and Setup Checking
```python
from execute.pymdp import validate_pymdp_simulation_setup

# Validate simulation setup before execution
validation_results = validate_pymdp_simulation_setup("simulation.py")

if validation_results['model_valid']:
    print("Model validation: ✅ PASSED")
else:
    print("Model validation: ❌ FAILED")
    for error in validation_results['errors']:
        print(f"  - {error}")

if validation_results['dependencies_available']:
    print("Dependencies: ✅ AVAILABLE")
else:
    print("Dependencies: ❌ MISSING")
    print(f"Missing: {validation_results['missing_deps']}")
```

---

## Active Inference Implementation

### Perception and Inference
- **State Estimation**: Variational inference for hidden state beliefs
- **Prediction Errors**: Precision-weighted prediction error minimization
- **Belief Updating**: Bayesian belief updating using observation likelihoods
- **Model Learning**: Parameter learning through experience

### Action and Planning
- **Policy Selection**: Expected free energy minimization
- **Action Selection**: Stochastic or deterministic action choice
- **Planning**: Multi-step planning with varying horizons
- **Exploration**: Intrinsic motivation through epistemic affordances

### Learning and Adaptation
- **Parameter Updates**: Gradient-based or sampling-based learning
- **Model Refinement**: Iterative model improvement
- **Adaptation**: Environment adaptation through learning
- **Generalization**: Transfer learning across similar environments

---

## Output Specification

### Output Products
- `pymdp_simulation_results.json` - Complete simulation results
- `belief_evolution_plots.png` - Belief trajectory visualizations
- `free_energy_analysis.json` - Free energy analysis results
- `performance_metrics.csv` - Performance metrics data
- `simulation_animation.gif` - Simulation animation (if enabled)

### Output Directory Structure
```
output/12_execute_output/
├── pymdp_simulation_results.json
├── belief_evolution_plots.png
├── free_energy_analysis.json
├── performance_metrics.csv
└── simulation_videos/
    └── trial_001.mp4
```

### Result Data Structure
```python
simulation_results = {
    'metadata': {
        'model_name': 'actinf_pomdp_agent',
        'framework': 'pymdp',
        'num_trials': 100,
        'trial_length': 50,
        'execution_time': 45.67,
        'timestamp': '2025-10-28T10:30:00Z'
    },
    'trials': [
        {
            'trial_id': 1,
            'observations': [...],
            'actions': [...],
            'beliefs': [...],
            'free_energy': [...],
            'rewards': [...]
        }
    ],
    'aggregate_metrics': {
        'avg_free_energy': -12.34,
        'avg_reward': 8.90,
        'convergence_rate': 0.95,
        'exploration_ratio': 0.23
    },
    'analysis': {
        'belief_accuracy': 0.87,
        'policy_entropy': 1.23,
        'learning_curves': [...]
    }
}
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 10-300 seconds per simulation (depends on complexity)
- **Memory**: 100-500MB depending on model size
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Model Loading**: < 1s
- **Inference Setup**: < 2s
- **Simulation Execution**: 5-250s (main bottleneck)
- **Result Analysis**: 2-50s
- **Visualization**: 1-20s

### Optimization Notes
- PyMDP inference is computationally intensive
- Memory usage scales with state/observation dimensions
- Parallel execution can significantly improve performance
- GPU acceleration available for some operations

---

## Error Handling

### Simulation Execution Errors
1. **Model Loading Failures**: Invalid PyMDP model files
2. **Inference Convergence Issues**: Poor model specification
3. **Memory Limitations**: Large models exceeding memory limits
4. **Dependency Issues**: Missing PyMDP or visualization libraries

### Recovery Strategies
- **Model Validation**: Comprehensive pre-execution validation
- **Parameter Adjustment**: Automatic parameter tuning for convergence
- **Memory Optimization**: Model size reduction or chunked execution
- **Fallback Execution**: Simplified execution with reduced features

### Error Examples
```python
try:
    results = execute_pymdp_simulation(model_path, config)
except PyMDPExecutionError as e:
    logger.error(f"PyMDP execution failed: {e}")
    # Attempt recovery with simplified config
    simplified_config = simplify_pymdp_config(config)
    results = execute_pymdp_simulation(model_path, simplified_config)
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `render.pymdp` - PyMDP code generation
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `execute.processor` - Main execution integration
- `tests.test_execute_pymdp*` - PyMDP execution tests

### Data Flow
```
PyMDP Code Generation → Model Validation → Simulation Setup → Inference Execution → Result Analysis → Visualization
```

---

## Testing

### Test Files
- `src/tests/test_execute_pymdp_integration.py` - Integration tests
- `src/tests/test_execute_pymdp_simulation.py` - Simulation tests
- `src/tests/test_execute_pymdp_analysis.py` - Analysis tests

### Test Coverage
- **Current**: 80%
- **Target**: 90%+

### Key Test Scenarios
1. PyMDP model loading and validation
2. Inference execution with various configurations
3. Simulation pipeline end-to-end testing
4. Result analysis and visualization
5. Error handling and recovery testing

### Test Commands
```bash
# Run PyMDP execution tests
pytest src/tests/test_execute_pymdp*.py -v

# Run with coverage
pytest src/tests/test_execute_pymdp*.py --cov=src/execute/pymdp --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `execute.run_pymdp_simulation` - Execute PyMDP simulation
- `execute.analyze_pymdp_results` - Analyze simulation results
- `execute.visualize_pymdp_beliefs` - Visualize belief trajectories
- `execute.validate_pymdp_setup` - Validate simulation setup

### Tool Endpoints
```python
@mcp_tool("execute.run_pymdp_simulation")
def run_pymdp_simulation_tool(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute PyMDP simulation with given configuration"""
    return execute_pymdp_simulation(model_path, config)
```

---

## Active Inference Metrics and Analysis

### Performance Metrics
- **Free Energy**: Variational free energy trajectory
- **Belief Accuracy**: Correctness of state inferences
- **Policy Entropy**: Exploration vs exploitation balance
- **Convergence Rate**: Speed of inference convergence
- **Reward Rate**: Average reward per trial

### Analysis Techniques
- **Time Series Analysis**: Belief and action trajectories
- **Statistical Testing**: Significance testing of results
- **Correlation Analysis**: Relationships between variables
- **Learning Curve Analysis**: Performance improvement over time

### Visualization Types
- **Belief Evolution**: Posterior belief trajectories over time
- **Action Selection**: Action probability distributions
- **Free Energy Landscape**: Free energy minimization paths
- **Reward Learning**: Reward accumulation curves

---

## Development Guidelines

### Adding New PyMDP Features
1. Update execution logic in `execute_pymdp.py`
2. Add new analysis functions in analysis modules
3. Update visualization capabilities
4. Add comprehensive tests

### Performance Optimization
- Profile inference bottlenecks
- Implement parallel trial execution
- Optimize memory usage for large models
- Use efficient data structures for results

---

## Troubleshooting

### Common Issues

#### Issue 1: "Inference not converging"
**Symptom**: PyMDP inference fails to converge within iteration limit
**Cause**: Poor model specification or inappropriate inference parameters
**Solution**: Adjust inference parameters (iterations, threshold) or improve model

#### Issue 2: "Memory usage too high"
**Symptom**: Simulation runs out of memory during execution
**Cause**: Model too large or too many trials
**Solution**: Reduce model size, decrease trial count, or use chunked execution

#### Issue 3: "Visualization fails"
**Symptom**: Result visualization generation fails
**Cause**: Missing plotting dependencies or invalid data
**Solution**: Install missing dependencies or validate result data

### Debug Mode
```python
# Enable debug output for PyMDP execution
results = execute_pymdp_simulation(model_path, config, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete PyMDP simulation execution pipeline
- Active Inference algorithm implementation
- Comprehensive result analysis and visualization
- Real-time simulation monitoring
- Extensive error handling and recovery
- MCP tool integration

**Known Limitations**:
- Large-scale simulations may require significant memory
- Complex hierarchical models need manual optimization
- Some advanced PyMDP features require custom implementation

### Roadmap
- **Next Version**: Enhanced parallel execution support
- **Future**: GPU acceleration for inference
- **Advanced**: Integration with PyMDP's latest features

---

## References

### Related Documentation
- [Execute Module](../../execute/AGENTS.md) - Parent execute module
- [PyMDP Render](../../render/pymdp/AGENTS.md) - PyMDP code generation
- [PyMDP Documentation](https://pymdp.readthedocs.io/) - Official PyMDP docs

### External Resources
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference)
- [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process)
- [Variational Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)

---

**Last Updated**: 2026-01-07
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready




