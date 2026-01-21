# PyMDP Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for PyMDP (Python Markov Decision Process) framework simulations from GNN specifications

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / PyMDP

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to PyMDP simulation code
2. Generate complete PyMDP agent implementations
3. Create executable Python scripts for Active Inference simulations
4. Handle PyMDP-specific optimizations and configurations
5. Support PyMDP's probabilistic programming paradigm

### Key Capabilities
- Complete PyMDP agent code generation
- Markov Decision Process modeling from GNN specifications
- Active Inference simulation setup
- PyMDP-specific template management
- Error handling and validation for PyMDP compatibility

---

## API Reference

### Exported Function from `__init__.py`

#### `render_gnn_to_pymdp(gnn_spec: Union[str, Path, Dict[str, Any]], output_path: Path, **kwargs) -> Tuple[bool, str, List[str]]`
**Description**: Main function exported from the module. Renders GNN specification to PyMDP simulation code.

**Parameters**:
- `gnn_spec` (Union[str, Path, Dict[str, Any]]): GNN specification (file path, content string, or parsed dictionary)
- `output_path` (Path): Output file path for generated PyMDP code
- `**kwargs`: Additional PyMDP generation options:
  - `optimize` (bool): Enable PyMDP-specific optimizations (default: True)
  - `include_examples` (bool): Include example usage code (default: True)

**Returns**: `Tuple[bool, str, List[str]]` - Tuple containing:
- `success` (bool): Whether rendering succeeded
- `message` (str): Status message
- `generated_files` (List[str]): List of generated file paths

**Location**: `src/render/pymdp/pymdp_renderer.py`

**Example**:
```python
from render.pymdp import render_gnn_to_pymdp
from pathlib import Path

# Render GNN file to PyMDP code
success, message, files = render_gnn_to_pymdp(
    gnn_spec=Path("input/model.md"),
    output_path=Path("output/simulation.py"),
    optimize=True,
    include_examples=True
)
```

### Internal Functions (Used by render_gnn_to_pymdp)

#### `generate_pymdp_code(model_data: Dict[str, Any], output_path: Optional[Union[str, Path]] = None, **kwargs) -> str`
**Description**: Internal function used by renderer to generate PyMDP simulation code from GNN model data.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data dictionary with variables, connections, matrices
- `output_path` (Optional[Union[str, Path]]): Output file path (optional, if provided code is also written to file)
- `optimize` (bool, optional): Enable PyMDP-specific optimizations (default: True)
- `include_examples` (bool, optional): Include example usage code (default: True)
- `**kwargs`: Additional PyMDP generation options

**Returns**: `str` - Generated PyMDP code as string

**Location**: `src/render/pymdp/pymdp_renderer.py` (internal implementation)

#### `convert_gnn_to_pymdp(model_data: Dict[str, Any], **kwargs) -> Dict[str, Any]`
**Description**: Convert GNN model data to PyMDP-compatible format.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data with variables, connections, matrices
- `validate` (bool, optional): Validate PyMDP compatibility (default: True)
- `**kwargs`: Additional conversion options

**Returns**: `Dict[str, Any]` - PyMDP-compatible model structure with:
- `state_space` (List): State space definition
- `observation_space` (List): Observation space definition
- `action_space` (List): Action space definition
- `matrices` (Dict): A, B, C, D matrices

#### `create_pymdp_agent(model_structure: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Create PyMDP agent implementation code.

**Parameters**:
- `model_structure` (Dict[str, Any]): PyMDP-compatible model structure
- `config` (Dict[str, Any], optional): PyMDP configuration options (default: {})
- `agent_type` (str, optional): Agent type ("standard", "optimized") (default: "standard")
- `**kwargs`: Additional agent generation options

**Returns**: `str` - PyMDP agent code as string

#### `generate_pymdp_simulation_script(model_data: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Generate complete PyMDP simulation script with agent and execution loop.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data
- `config` (Dict[str, Any], optional): Simulation configuration (default: {})
- `num_steps` (int, optional): Number of simulation steps (default: 100)
- `include_visualization` (bool, optional): Include visualization code (default: True)
- `**kwargs`: Additional simulation options

**Returns**: `str` - Complete simulation script as string

---

## Dependencies

### Required Dependencies
- `pymdp` - PyMDP probabilistic programming library
- `numpy` - Numerical computing
- `pathlib` - Path manipulation

### Optional Dependencies
- `matplotlib` - Visualization support (fallback: no plotting)
- `scipy` - Advanced mathematical functions (fallback: numpy-only)

### Internal Dependencies
- `render.renderer` - Base rendering functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### PyMDP Configuration
```python
PYMDP_CONFIG = {
    'inference_algorithm': 'VMP',      # Variational Message Passing
    'learning_rate': 0.1,             # Learning rate for inference
    'num_iterations': 100,            # Maximum inference iterations
    'convergence_threshold': 1e-6,    # Convergence threshold
    'action_selection': 'stochastic', # Action selection method
    'planning_horizon': 1,            # Planning horizon
    'use_posterior': True,            # Use posterior for action selection
    'save_results': True              # Save simulation results
}
```

### Model Conversion Configuration
```python
CONVERSION_CONFIG = {
    'state_mapping': 'direct',        # How to map GNN states to PyMDP
    'observation_mapping': 'categorical',  # Observation representation
    'action_mapping': 'discrete',     # Action space mapping
    'reward_function': 'custom',      # Reward function type
    'transition_model': 'learnable',  # Transition model type
    'policy_type': 'random'           # Initial policy type
}
```

### Simulation Configuration
```python
SIMULATION_CONFIG = {
    'num_trials': 100,                # Number of simulation trials
    'trial_length': 50,               # Length of each trial
    'save_interval': 10,              # Save results every N trials
    'plot_results': True,             # Generate result plots
    'verbose': False,                 # Verbose output
    'random_seed': 42                 # Random seed for reproducibility
}
```

---

## Usage Examples

### Basic PyMDP Code Generation
```python
from render.pymdp import generate_pymdp_code

# Example GNN model data
model_data = {
    "variables": {
        "observations": {"domain": ["left", "right"], "type": "categorical"},
        "states": {"domain": ["start", "middle", "end"], "type": "categorical"},
        "actions": {"domain": ["move_left", "move_right"], "type": "categorical"}
    },
    "connections": [
        {"from": "states", "to": "observations", "type": "generative"},
        {"from": "actions", "to": "states", "type": "transition"}
    ]
}

# Generate PyMDP code
pymdp_code = generate_pymdp_code(model_data)
print(pymdp_code[:500])  # Print first 500 characters
```

### Complete Simulation Script Generation
```python
from render.pymdp import generate_pymdp_simulation_script

# Configuration for simulation
config = {
    'inference_algorithm': 'VMP',
    'num_trials': 50,
    'trial_length': 20,
    'save_results': True
}

# Generate complete script
simulation_script = generate_pymdp_simulation_script(model_data, config)

# Save to file
with open("pymdp_simulation.py", "w") as f:
    f.write(simulation_script)
```

### Model Conversion
```python
from render.pymdp import convert_gnn_to_pymdp

# Convert GNN model to PyMDP format
pymdp_model = convert_gnn_to_pymdp(model_data)

print("PyMDP Model Structure:")
print(f"A (observations): {pymdp_model['A'].shape}")
print(f"B (transitions): {pymdp_model['B'].shape}")
print(f"C (preferences): {pymdp_model['C'].shape}")
print(f"D (priors): {pymdp_model['D'].shape}")
```

### Agent Creation
```python
from render.pymdp import create_pymdp_agent

# Create PyMDP agent
agent_config = {
    'policy_type': 'random',
    'inference_params': {'num_iter': 10, 'threshold': 1e-4}
}

agent_code = create_pymdp_agent(model_data, agent_config)

# This generates the agent initialization code
print(agent_code)
```

---

## PyMDP Concepts Mapping

### GNN to PyMDP Mapping
- **Variables → States/Observations**: GNN variables become PyMDP state or observation factors
- **Connections → Likelihoods/Transitions**: Generative connections become likelihood arrays (A), transition connections become transition arrays (B)
- **Constraints → Preferences/Priors**: Constraints become preference arrays (C) or prior arrays (D)

### PyMDP Arrays Generated
- **A arrays**: Observation likelihoods P(o|s) - how likely observations are given states
- **B arrays**: Transition likelihoods P(s'|s,a) - state transitions given actions
- **C arrays**: Preference distributions - desired observations
- **D arrays**: Prior distributions - initial state beliefs

---

## Output Specification

### Output Products
- `*_pymdp_simulation.py` - Complete PyMDP simulation scripts
- `*_pymdp_model.py` - PyMDP model definition files
- `*_pymdp_agent.py` - PyMDP agent implementation files
- `pymdp_config.json` - PyMDP configuration file

### Output Directory Structure
```
output/11_render_output/
├── model_name_pymdp_simulation.py
├── model_name_pymdp_model.py
├── model_name_pymdp_agent.py
└── pymdp_config.json
```

### Generated Script Structure
```python
# Generated PyMDP simulation script structure
import pymdp
import numpy as np
from pymdp import utils

# Model definition
A = [...]  # Observation likelihoods
B = [...]  # Transition likelihoods
C = [...]  # Preferences
D = [...]  # Priors

# Agent creation
agent = pymdp.Agent(A=A, B=B, C=C, D=D)

# Simulation loop
for trial in range(num_trials):
    for t in range(trial_length):
        # Inference and action selection
        # Environment interaction
        # Learning updates

# Results and plotting
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 1-2 seconds per model
- **Memory**: 50-200MB depending on model complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Code Generation**: < 500ms
- **Template Processing**: < 1s
- **Validation**: < 500ms
- **File Writing**: < 100ms

### Optimization Notes
- PyMDP code generation is typically fast due to template-based approach
- Memory usage depends on model size and array dimensions
- Generated code is optimized for PyMDP's inference algorithms

---

## Error Handling

### PyMDP Generation Errors
1. **Invalid Model Structure**: GNN model cannot be mapped to PyMDP
2. **Array Dimension Mismatches**: Incompatible array sizes
3. **Configuration Errors**: Invalid PyMDP parameters

### Recovery Strategies
- **Model Validation**: Comprehensive pre-generation validation
- **Fallback Templates**: Use simpler templates for complex models
- **Error Reporting**: Detailed error messages with suggestions

### Error Examples
```python
try:
    pymdp_code = generate_pymdp_code(model_data)
except PyMDPGenerationError as e:
    logger.error(f"PyMDP generation failed: {e}")
    # Fallback to basic template
    pymdp_code = generate_basic_pymdp_template(model_data)
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/render/` (Step 11)
- **Main Script**: `11_render.py`

### Imports From
- `render.renderer` - Base rendering functionality
- `gnn.parsers` - GNN parsing and validation

### Imported By
- `render.processor` - Main render processing integration
- `execute.pymdp` - PyMDP execution module
- `tests.test_render_pymdp*` - PyMDP-specific tests

### Data Flow
```
GNN Model → PyMDP Conversion → Template Application → Code Generation → Validation → Output
```

---

## Testing

### Test Files
- `src/tests/test_render_pymdp_integration.py` - Integration tests
- `src/tests/test_render_pymdp_generation.py` - Code generation tests
- `src/tests/test_render_pymdp_validation.py` - Validation tests

### Test Coverage
- **Current**: 85%
- **Target**: 90%+

### Key Test Scenarios
1. PyMDP code generation from various GNN models
2. Generated code syntax and import validation
3. PyMDP array generation and dimension checking
4. Template application and customization
5. Error handling for invalid models

### Test Commands
```bash
# Run PyMDP-specific tests
pytest src/tests/test_render_pymdp*.py -v

# Run with coverage
pytest src/tests/test_render_pymdp*.py --cov=src/render/pymdp --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `render.generate_pymdp` - Generate PyMDP simulation code
- `render.convert_to_pymdp` - Convert GNN to PyMDP format
- `render.validate_pymdp` - Validate PyMDP model structure

### Tool Endpoints
```python
@mcp_tool("render.generate_pymdp")
def generate_pymdp_tool(model_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate PyMDP simulation code from GNN model"""
    return generate_pymdp_code(model_data, **config)
```

---

## PyMDP-Specific Features

### Active Inference Implementation
- **Perception**: State estimation using observation likelihoods (A arrays)
- **Action Selection**: Policy selection based on expected free energy
- **Learning**: Model parameter updates through experience
- **Planning**: Multi-step planning for optimal behavior

### Optimization Strategies
- **Sparse Representations**: Efficient sparse matrix representations
- **Vectorized Operations**: NumPy vectorization for performance
- **Caching**: Result caching for repeated computations
- **Parallel Processing**: Parallel inference when possible

---

## Development Guidelines

### Adding New PyMDP Features
1. Update model conversion logic in `pymdp_converter.py`
2. Add new templates in `pymdp_templates.py`
3. Update configuration options
4. Add comprehensive tests

### Template Management
- Templates are stored in `pymdp_templates.py`
- Use Jinja2-style templating for dynamic code generation
- Maintain template modularity for different PyMDP features
- Include error handling in generated code

---

## Troubleshooting

### Common Issues

#### Issue 1: "Array dimension mismatch in PyMDP model"
**Symptom**: PyMDP model creation fails with dimension errors
**Cause**: Incompatible GNN variable domains or connection structures
**Solution**: Validate GNN model structure before conversion

#### Issue 2: "PyMDP inference not converging"
**Symptom**: Generated simulations fail to converge
**Cause**: Poor model specification or inappropriate inference parameters
**Solution**: Adjust inference parameters or simplify model

#### Issue 3: "Memory error during simulation"
**Symptom**: Large models cause memory allocation failures
**Cause**: Model too complex for available memory
**Solution**: Reduce model complexity or use sparse representations

### Debug Mode
```python
# Enable debug output for PyMDP generation
result = generate_pymdp_code(model_data, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete PyMDP code generation pipeline
- Active Inference model implementation
- Template-based code generation
- Comprehensive error handling and validation
- MCP tool integration

**Known Limitations**:
- Complex hierarchical models may require manual optimization
- Very large state spaces may impact performance
- Some advanced PyMDP features require manual implementation

### Roadmap
- **Next Version**: Enhanced hierarchical model support
- **Future**: Automatic model optimization
- **Advanced**: Integration with PyMDP's latest features

---

## References

### Related Documentation
- [Render Module](../../render/AGENTS.md) - Parent render module
- [PyMDP Documentation](https://pymdp.readthedocs.io/) - Official PyMDP docs
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) - Active Inference theory

### External Resources
- [Markov Decision Processes](https://en.wikipedia.org/wiki/Markov_decision_process)
- [Probabilistic Programming](https://en.wikipedia.org/wiki/Probabilistic_programming)
- [PyMDP GitHub](https://github.com/infer-actively/pymdp)

---

**Last Updated**: 2026-01-21
**Maintainer**: Render Module Team
**Status**: ✅ Production Ready




