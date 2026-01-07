# ActiveInference.jl Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for ActiveInference.jl framework simulations from GNN specifications

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / ActiveInference.jl

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to ActiveInference.jl simulation code
2. Generate Active Inference agent implementations in Julia
3. Create hierarchical and temporal Active Inference models
4. Handle ActiveInference.jl-specific optimizations and configurations
5. Support complex multi-level Active Inference architectures

### Key Capabilities
- Complete Active Inference agent code generation
- Hierarchical model implementation
- Temporal dynamics and planning
- Multi-agent scenarios support
- ActiveInference.jl-specific template management
- Error handling and validation for Active Inference compatibility

---

## API Reference

### Public Functions

#### `generate_activeinference_jl_code(model_data: Dict[str, Any], output_path: Optional[Union[str, Path]] = None, **kwargs) -> str`
**Description**: Generate ActiveInference.jl simulation code from GNN model data.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data dictionary with variables, connections, matrices
- `output_path` (Optional[Union[str, Path]]): Output file path (optional, if provided code is also written to file)
- `hierarchical` (bool, optional): Enable hierarchical model support (default: False)
- `temporal` (bool, optional): Enable temporal dynamics (default: True)
- `**kwargs`: Additional ActiveInference.jl generation options

**Returns**: `str` - Generated ActiveInference.jl code as string

**Example**:
```python
from render.activeinference_jl import generate_activeinference_jl_code
from pathlib import Path

# Generate ActiveInference.jl code
ai_code = generate_activeinference_jl_code(
    model_data=parsed_gnn_model,
    output_path=Path("output/active_inference_simulation.jl"),
    hierarchical=True,
    temporal=True
)

# Code is also saved to file if output_path provided
```

#### `convert_gnn_to_activeinference(model_data: Dict[str, Any], **kwargs) -> Dict[str, Any]`
**Description**: Convert GNN model data to ActiveInference.jl-compatible format.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data with variables, connections, matrices
- `validate` (bool, optional): Validate ActiveInference.jl compatibility (default: True)
- `**kwargs`: Additional conversion options

**Returns**: `Dict[str, Any]` - ActiveInference.jl-compatible model structure with:
- `agent_structure` (Dict): Agent definition
- `generative_model` (Dict): Generative model specification
- `inference_config` (Dict): Inference algorithm configuration
- `hierarchical_levels` (List[Dict]): Hierarchical levels if applicable

#### `create_activeinference_agent(model_structure: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Create ActiveInference.jl agent implementation code.

**Parameters**:
- `model_structure` (Dict[str, Any]): ActiveInference.jl-compatible model structure
- `config` (Dict[str, Any], optional): ActiveInference.jl configuration options (default: {})
- `agent_type` (str, optional): Agent type ("standard", "hierarchical", "temporal") (default: "standard")
- `**kwargs`: Additional agent generation options

**Returns**: `str` - ActiveInference.jl agent code as string

#### `generate_activeinference_simulation_script(model_data: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Generate complete ActiveInference.jl simulation script with agent and execution.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data
- `config` (Dict[str, Any], optional): Simulation configuration (default: {})
- `num_steps` (int, optional): Number of simulation steps (default: 100)
- `include_analysis` (bool, optional): Include analysis code (default: True)
- `**kwargs`: Additional simulation options

**Returns**: `str` - Complete simulation script as string

---

## Dependencies

### Required Dependencies
- `ActiveInference.jl` - ActiveInference.jl Julia package
- `Julia` - Julia programming language runtime
- `Distributions.jl` - Probability distributions
- `Plots.jl` - Visualization support

### Optional Dependencies
- `Agents.jl` - Agent-based modeling (fallback: single agent)
- `DifferentialEquations.jl` - Continuous dynamics (fallback: discrete time)

### Internal Dependencies
- `render.renderer` - Base rendering functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### ActiveInference.jl Configuration
```python
ACTIVEINFERENCE_CONFIG = {
    'agent_type': 'single',           # Single or multi-agent
    'planning_horizon': 5,            # Planning horizon
    'time_step': 0.1,                 # Time step for continuous models
    'inference_method': 'variational', # Inference method
    'action_selection': 'stochastic', # Action selection strategy
    'learning_rate': 0.01,            # Learning rate
    'exploration_bonus': 0.1,         # Exploration parameter
    'save_history': True,             # Save simulation history
    'parallel_execution': False       # Parallel processing
}
```

### Model Conversion Configuration
```python
CONVERSION_CONFIG = {
    'hierarchy_levels': 1,            # Number of hierarchical levels
    'temporal_depth': 10,             # Temporal planning depth
    'state_representation': 'continuous',  # State representation type
    'action_space': 'discrete',       # Action space type
    'observation_model': 'linear',    # Observation model type
    'transition_model': 'nonlinear'   # Transition model type
}
```

### Simulation Configuration
```python
SIMULATION_CONFIG = {
    'total_time': 100.0,              # Total simulation time
    'time_step': 0.1,                 # Simulation time step
    'num_trials': 10,                 # Number of simulation trials
    'initial_conditions': 'random',   # Initial condition generation
    'boundary_conditions': 'periodic', # Boundary condition handling
    'output_frequency': 1,            # Output data frequency
    'visualization': True             # Generate visualizations
}
```

---

## Usage Examples

### Basic ActiveInference.jl Code Generation
```python
from render.activeinference_jl import generate_activeinference_jl_code

# Example GNN model data for Active Inference
model_data = {
    "variables": {
        "internal_states": {"domain": "continuous", "dimensions": 3},
        "observations": {"domain": "continuous", "dimensions": 2},
        "actions": {"domain": "discrete", "values": ["left", "right", "forward"]},
        "policies": {"domain": "discrete", "values": ["explore", "exploit"]}
    },
    "connections": [
        {"from": "internal_states", "to": "observations", "type": "generative"},
        {"from": "actions", "to": "internal_states", "type": "transition"},
        {"from": "policies", "to": "actions", "type": "control"}
    ],
    "parameters": {
        "free_energy_precision": 1.0,
        "temporal_horizon": 5,
        "learning_rate": 0.01
    }
}

# Generate ActiveInference.jl code
ai_code = generate_activeinference_jl_code(model_data)
print(ai_code[:500])  # Print first 500 characters
```

### Complete Simulation Script Generation
```python
from render.activeinference_jl import generate_activeinference_simulation_script

# Configuration for Active Inference simulation
config = {
    'agent_type': 'single',
    'planning_horizon': 3,
    'total_time': 50.0,
    'learning_rate': 0.05,
    'visualization': True
}

# Generate complete script
simulation_script = generate_activeinference_simulation_script(model_data, config)

# Save to file
with open("activeinference_simulation.jl", "w") as f:
    f.write(simulation_script)
```

### Model Conversion
```python
from render.activeinference_jl import convert_gnn_to_activeinference

# Convert GNN model to ActiveInference.jl format
ai_model = convert_gnn_to_activeinference(model_data)

print("ActiveInference.jl Model Structure:")
print(f"States: {ai_model['states']['dimensions']}D")
print(f"Observations: {ai_model['observations']['dimensions']}D")
print(f"Actions: {len(ai_model['actions']['values'])} discrete actions")
print(f"Hierarchy levels: {ai_model['hierarchy_levels']}")
```

### Agent Creation
```python
from render.activeinference_jl import create_activeinference_agent

# Create ActiveInference.jl agent
agent_config = {
    'inference_method': 'variational',
    'planning_horizon': 5,
    'action_selection': 'stochastic'
}

agent_code = create_activeinference_agent(model_data, agent_config)

# This generates the agent initialization and inference code
print(agent_code)
```

---

## Active Inference Concepts Mapping

### GNN to ActiveInference.jl Mapping
- **Variables → States/Observations**: GNN variables become internal states or observations
- **Connections → Generative Models**: Generative connections become likelihood models
- **Actions → Control**: Action variables become control inputs
- **Policies → Planning**: Policy variables become planning mechanisms
- **Parameters → Free Energy**: Model parameters become precision parameters

### ActiveInference.jl Components Generated
- **Agent Structure**: Agent initialization with states, observations, actions
- **Generative Model**: Probabilistic generative model specification
- **Inference Engine**: Variational or sampling-based inference
- **Planning System**: Multi-step planning and policy selection
- **Learning Rules**: Parameter learning and adaptation

---

## Output Specification

### Output Products
- `*_activeinference_simulation.jl` - Complete ActiveInference.jl simulation scripts
- `*_activeinference_agent.jl` - ActiveInference.jl agent implementation files
- `*_activeinference_model.jl` - Model definition files
- `activeinference_config.jl` - Configuration file

### Output Directory Structure
```
output/11_render_output/
├── model_name_activeinference_simulation.jl
├── model_name_activeinference_agent.jl
├── model_name_activeinference_model.jl
└── activeinference_config.jl
```

### Generated Script Structure
```julia
# Generated ActiveInference.jl simulation script structure
using ActiveInference
using Distributions
using Plots

# Agent parameters
parameters = (;
    # State and observation dimensions
    num_states = 3,
    num_observations = 2,
    num_actions = 3,

    # Generative model parameters
    A = [...],  # Observation likelihoods
    B = [...],  # Transition likelihoods
    C = [...],  # Preferences
    D = [...],  # Priors

    # Inference parameters
    planning_horizon = 5,
    learning_rate = 0.01,
    exploration_bonus = 0.1
)

# Agent initialization
agent = init_agent(parameters)

# Simulation loop
for t in 1:total_time
    # Inference step
    infer_states!(agent, observation)

    # Planning step
    plan_actions!(agent)

    # Action selection
    action = select_action(agent)

    # Environment interaction
    next_observation = environment_step(action)

    # Learning step
    update_parameters!(agent, action, next_observation)
end

# Results visualization
plot_simulation_results(agent)
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 2-3 seconds per model
- **Memory**: 150-400MB depending on model complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Code Generation**: < 1s
- **Template Processing**: < 2s
- **Validation**: < 1s
- **File Writing**: < 100ms

### Optimization Notes
- ActiveInference.jl generation includes hierarchical optimizations
- Memory usage depends on state dimensions and planning horizon
- Generated code is optimized for Active Inference algorithms

---

## Error Handling

### ActiveInference.jl Generation Errors
1. **Invalid Model Structure**: GNN model cannot be mapped to Active Inference
2. **Dimension Mismatches**: Incompatible state/observation dimensions
3. **Julia Syntax Errors**: Generated code has syntax issues

### Recovery Strategies
- **Model Validation**: Comprehensive pre-generation validation
- **Dimension Checking**: Automatic dimension compatibility verification
- **Template Fallback**: Use simpler templates for complex models

### Error Examples
```python
try:
    ai_code = generate_activeinference_jl_code(model_data)
except ActiveInferenceGenerationError as e:
    logger.error(f"ActiveInference.jl generation failed: {e}")
    # Fallback to basic template
    ai_code = generate_basic_activeinference_template(model_data)
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
- `execute.activeinference_jl` - ActiveInference.jl execution module
- `tests.test_render_activeinference*` - ActiveInference.jl-specific tests

### Data Flow
```
GNN Model → ActiveInference.jl Conversion → Template Application → Julia Code Generation → Validation → Output
```

---

## Testing

### Test Files
- `src/tests/test_render_activeinference_integration.py` - Integration tests
- `src/tests/test_render_activeinference_generation.py` - Code generation tests
- `src/tests/test_render_activeinference_validation.py` - Validation tests

### Test Coverage
- **Current**: 74%
- **Target**: 80%+

### Key Test Scenarios
1. ActiveInference.jl code generation from various GNN models
2. Generated Julia code syntax and import validation
3. Hierarchical model structure generation
4. Temporal planning implementation
5. Error handling for invalid models

### Test Commands
```bash
# Run ActiveInference.jl-specific tests
pytest src/tests/test_render_activeinference*.py -v

# Run with coverage
pytest src/tests/test_render_activeinference*.py --cov=src/render/activeinference_jl --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `render.generate_activeinference` - Generate ActiveInference.jl simulation code
- `render.convert_to_activeinference` - Convert GNN to ActiveInference.jl format
- `render.validate_activeinference` - Validate ActiveInference.jl model structure

### Tool Endpoints
```python
@mcp_tool("render.generate_activeinference")
def generate_activeinference_tool(model_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate ActiveInference.jl simulation code from GNN model"""
    return generate_activeinference_jl_code(model_data, **config)
```

---

## Active Inference-Specific Features

### Hierarchical Active Inference Implementation
- **Multi-level Planning**: Hierarchical policy selection
- **Temporal Abstraction**: Different timescales for different levels
- **Goal-directed Behavior**: Preference-driven action selection
- **Adaptive Precision**: Context-dependent uncertainty handling
- **Meta-learning**: Learning to learn in changing environments

### Optimization Strategies
- **Message Passing**: Efficient belief propagation
- **Gradient-based Learning**: Automatic differentiation for parameter updates
- **Sparse Representations**: Memory-efficient state representations
- **Parallel Processing**: Parallel inference across hierarchy levels

---

## Development Guidelines

### Adding New ActiveInference.jl Features
1. Update model conversion logic in `activeinference_jl_renderer.py`
2. Add new templates in renderer files
3. Update configuration options for new features
4. Add comprehensive tests

### Template Management
- Templates are implemented as functions in renderer files
- Use string templating for dynamic Julia code generation
- Maintain template modularity for different Active Inference features
- Include proper Julia imports and error handling

---

## Troubleshooting

### Common Issues

#### Issue 1: "Hierarchical model structure invalid"
**Symptom**: ActiveInference.jl model creation fails with structure errors
**Cause**: Incompatible GNN hierarchy or connection patterns
**Solution**: Validate GNN model hierarchical structure before conversion

#### Issue 2: "Planning horizon too large for model"
**Symptom**: Simulation fails due to memory or computation limits
**Cause**: Planning horizon too large for model complexity
**Solution**: Reduce planning horizon or simplify model

#### Issue 3: "Free energy computation not converging"
**Symptom**: Inference fails to converge within iterations
**Cause**: Poor model specification or inappropriate parameters
**Solution**: Adjust inference parameters or model structure

### Debug Mode
```python
# Enable debug output for ActiveInference.jl generation
result = generate_activeinference_jl_code(model_data, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete ActiveInference.jl code generation pipeline
- Hierarchical Active Inference model implementation
- Temporal planning and multi-level control
- Comprehensive error handling and validation
- MCP tool integration

**Known Limitations**:
- Complex multi-agent scenarios require manual optimization
- Very large hierarchical models may impact performance
- Some advanced Active Inference features require manual implementation

### Roadmap
- **Next Version**: Enhanced multi-agent support
- **Future**: Automatic model structure optimization
- **Advanced**: Integration with latest Active Inference research

---

## References

### Related Documentation
- [Render Module](../../render/AGENTS.md) - Parent render module
- [ActiveInference.jl](https://github.com/ilabcode/ActiveInference.jl) - ActiveInference.jl package
- [Active Inference](https://en.wikipedia.org/wiki/Active_inference) - Active Inference theory

### External Resources
- [Active Inference Theory](https://www.sciencedirect.com/science/article/pii/S0022249615000164)
- [Free Energy Principle](https://en.wikipedia.org/wiki/Free_energy_principle)
- [Hierarchical Reinforcement Learning](https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning)

---

**Last Updated**: 2026-01-07
**Maintainer**: Render Module Team
**Status**: ✅ Production Ready




