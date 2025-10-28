# JAX Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for JAX framework simulations with high-performance computing from GNN specifications

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / JAX

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to JAX-optimized simulation code
2. Generate JIT-compiled, differentiable simulations
3. Create vectorized implementations with automatic differentiation
4. Handle JAX-specific optimizations and GPU acceleration
5. Support high-performance probabilistic programming

### Key Capabilities
- JIT-compiled simulation code generation
- Automatic differentiation implementation
- GPU acceleration support
- Vectorized operations for performance
- JAX-specific template management
- Memory-efficient implementations

---

## API Reference

### Public Functions

#### `generate_jax_code(model_data: Dict[str, Any], output_path: Optional[str] = None) -> str`
**Description**: Generate JAX simulation code from GNN model data

**Parameters**:
- `model_data` (Dict): GNN model data dictionary
- `output_path` (Optional[str]): Output file path (optional)

**Returns**: Generated JAX code as string

**Example**:
```python
from render.jax import generate_jax_code

# Generate JAX code
jax_code = generate_jax_code(model_data)

# Save to file
with open("jax_simulation.py", "w") as f:
    f.write(jax_code)
```

#### `convert_gnn_to_jax(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Convert GNN model data to JAX-compatible format

**Parameters**:
- `model_data` (Dict): GNN model data

**Returns**: JAX-compatible model structure

#### `create_jax_simulation(model_structure: Dict[str, Any], config: Dict[str, Any]) -> str`
**Description**: Create JAX simulation implementation

**Parameters**:
- `model_structure` (Dict): Model structure data
- `config` (Dict): JAX configuration options

**Returns**: JAX simulation code

#### `generate_jax_optimized_code(model_data: Dict[str, Any], config: Dict[str, Any]) -> str`
**Description**: Generate optimized JAX code with performance enhancements

**Parameters**:
- `model_data` (Dict): GNN model data
- `config` (Dict): Optimization configuration

**Returns**: Optimized JAX code

---

## Dependencies

### Required Dependencies
- `jax` - JAX library for high-performance computing
- `jaxlib` - JAX compilation backend
- `numpy` - Array operations (JAX-compatible)
- `jax.numpy` - JAX numpy operations

### Optional Dependencies
- `optax` - Gradient-based optimization (fallback: basic SGD)
- `flax` - Neural network library (fallback: basic operations)
- `distrax` - Probability distributions (fallback: basic distributions)

### Internal Dependencies
- `render.renderer` - Base rendering functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### JAX Configuration
```python
JAX_CONFIG = {
    'jit_compile': True,              # Enable JIT compilation
    'enable_x64': True,               # Enable 64-bit precision
    'gpu_acceleration': True,         # Enable GPU acceleration
    'vectorize_operations': True,     # Vectorize operations
    'gradient_computation': True,     # Enable automatic differentiation
    'parallel_execution': False,      # Enable parallel execution
    'memory_optimization': True,      # Optimize memory usage
    'debug_mode': False               # Enable debug features
}
```

### Model Conversion Configuration
```python
CONVERSION_CONFIG = {
    'state_representation': 'array',  # State representation type
    'parameter_format': 'pytree',     # Parameter organization
    'computation_graph': 'functional', # Computation structure
    'differentiation_mode': 'reverse', # AD mode (forward/reverse)
    'vectorization_level': 'batch',   # Vectorization strategy
    'memory_layout': 'contiguous',    # Memory layout optimization
    'precision': 'float32'            # Numerical precision
}
```

### Simulation Configuration
```python
SIMULATION_CONFIG = {
    'batch_size': 32,                 # Batch size for vectorization
    'num_steps': 1000,                # Simulation steps
    'learning_rate': 0.01,            # Learning rate for optimization
    'optimizer': 'adam',              # Optimization algorithm
    'loss_function': 'mse',           # Loss function
    'convergence_threshold': 1e-6,    # Convergence criterion
    'checkpoint_frequency': 100,      # Checkpoint saving frequency
    'performance_monitoring': True    # Enable performance tracking
}
```

---

## Usage Examples

### Basic JAX Code Generation
```python
from render.jax import generate_jax_code

# Example GNN model data for JAX optimization
model_data = {
    "variables": {
        "states": {"domain": "continuous", "dimensions": 4, "dtype": "float32"},
        "observations": {"domain": "continuous", "dimensions": 2, "dtype": "float32"},
        "actions": {"domain": "continuous", "dimensions": 1, "dtype": "float32"},
        "parameters": {"domain": "continuous", "dimensions": 10, "dtype": "float32"}
    },
    "connections": [
        {"from": "states", "to": "observations", "type": "linear"},
        {"from": "actions", "to": "states", "type": "transition"},
        {"from": "parameters", "to": "states", "type": "modulation"}
    ],
    "optimization": {
        "loss": "prediction_error",
        "method": "gradient_descent",
        "batch_size": 64
    }
}

# Generate JAX code
jax_code = generate_jax_code(model_data)
print(jax_code[:500])  # Print first 500 characters
```

### Optimized Simulation Generation
```python
from render.jax import generate_jax_optimized_code

# Configuration for optimized JAX simulation
config = {
    'jit_compile': True,
    'gpu_acceleration': True,
    'batch_size': 128,
    'gradient_computation': True,
    'performance_monitoring': True
}

# Generate optimized code
optimized_code = generate_jax_optimized_code(model_data, config)

# Save to file
with open("optimized_jax_simulation.py", "w") as f:
    f.write(optimized_code)
```

### Model Conversion
```python
from render.jax import convert_gnn_to_jax

# Convert GNN model to JAX format
jax_model = convert_gnn_to_jax(model_data)

print("JAX Model Structure:")
print(f"States: {jax_model['states']['shape']}")
print(f"Parameters: {jax_model['parameters']['shape']}")
print(f"Batch size: {jax_model['batch_size']}")
print(f"JIT compiled: {jax_model['jit_compiled']}")
```

### Gradient-based Learning
```python
from render.jax import create_jax_simulation

# Create simulation with gradient-based learning
sim_config = {
    'learning_enabled': True,
    'optimizer': 'adam',
    'loss_function': 'mse',
    'gradient_clipping': 1.0
}

simulation_code = create_jax_simulation(model_data, sim_config)

# This generates code for differentiable simulation
print(simulation_code)
```

---

## JAX Performance Features Mapping

### GNN to JAX Mapping
- **Variables → Arrays**: GNN variables become JAX arrays with shapes
- **Connections → Functions**: Relationships become differentiable functions
- **Parameters → Trainable**: Model parameters become trainable variables
- **Optimization → Gradients**: Learning becomes gradient-based optimization
- **Computation → JIT**: Operations become JIT-compiled for performance

### JAX Components Generated
- **JIT Functions**: @jax.jit decorated functions for performance
- **Grad Functions**: jax.grad for automatic differentiation
- **VMap Functions**: jax.vmap for vectorized operations
- **PyTree Structures**: Nested parameter structures
- **Random Keys**: JAX random number generation

---

## Output Specification

### Output Products
- `*_jax_simulation.py` - Complete JAX simulation scripts
- `*_jax_model.py` - JAX model definition files
- `*_jax_training.py` - Training and optimization code
- `jax_config.json` - JAX configuration file

### Output Directory Structure
```
output/11_render_output/
├── model_name_jax_simulation.py
├── model_name_jax_model.py
├── model_name_jax_training.py
└── jax_config.json
```

### Generated Script Structure
```python
# Generated JAX simulation script structure
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Enable 64-bit precision if needed
jax.config.update("jax_enable_x64", True)

# Model parameters (as PyTree)
params = {
    'transition_matrix': jnp.zeros((4, 4)),
    'observation_matrix': jnp.zeros((2, 4)),
    'bias_terms': jnp.zeros(4),
    'noise_parameters': jnp.ones(2) * 0.1
}

# JIT-compiled simulation step
@jit
def simulation_step(state, action, params, key):
    # Transition model
    new_state = jnp.dot(params['transition_matrix'], state) + action * params['bias_terms']

    # Add noise
    noise = random.normal(key, shape=(4,)) * params['noise_parameters'][0]
    new_state = new_state + noise

    # Observation model
    observation = jnp.dot(params['observation_matrix'], new_state)
    obs_noise = random.normal(key, shape=(2,)) * params['noise_parameters'][1]
    observation = observation + obs_noise

    return new_state, observation

# Loss function
def loss_fn(params, states, actions, observations):
    # Simulate trajectory
    # Compute prediction error
    return prediction_error

# Gradient-based optimization
grad_fn = grad(loss_fn)
optimizer = optax.adam(learning_rate=0.01)

# Training loop
for step in range(num_steps):
    grads = grad_fn(params, batch_states, batch_actions, batch_observations)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

# Vectorized evaluation
batch_simulate = vmap(simulation_step, in_axes=(0, 0, None, 0))
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 2-4 seconds per model
- **Memory**: 200-500MB depending on model complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Code Generation**: < 1s
- **JIT Compilation**: 1-3s (first run)
- **Template Processing**: < 1s
- **Validation**: < 500ms

### Optimization Notes
- JAX generation includes automatic differentiation setup
- Memory usage depends on batch size and model dimensions
- Generated code is optimized for GPU acceleration

---

## Error Handling

### JAX Generation Errors
1. **Shape Mismatch**: Incompatible array dimensions
2. **Dtype Issues**: Unsupported data types for operations
3. **JIT Compilation Errors**: Code that cannot be JIT-compiled

### Recovery Strategies
- **Shape Validation**: Comprehensive dimension checking before generation
- **Type Conversion**: Automatic dtype conversion when possible
- **Fallback Implementation**: Non-JIT version when compilation fails

### Error Examples
```python
try:
    jax_code = generate_jax_code(model_data)
except JAXGenerationError as e:
    logger.error(f"JAX generation failed: {e}")
    # Fallback to basic template
    jax_code = generate_basic_jax_template(model_data)
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
- `execute.jax` - JAX execution module
- `tests.test_render_jax*` - JAX-specific tests

### Data Flow
```
GNN Model → JAX Conversion → JIT Compilation Setup → Gradient Function Creation → Optimization Setup → Validation → Output
```

---

## Testing

### Test Files
- `src/tests/test_render_jax_integration.py` - Integration tests
- `src/tests/test_render_jax_generation.py` - Code generation tests
- `src/tests/test_render_jax_performance.py` - Performance tests

### Test Coverage
- **Current**: 76%
- **Target**: 85%+

### Key Test Scenarios
1. JAX code generation from various GNN models
2. Generated Python code syntax and import validation
3. JIT compilation success and performance
4. Gradient computation correctness
5. GPU acceleration functionality
6. Error handling for invalid JAX operations

### Test Commands
```bash
# Run JAX-specific tests
pytest src/tests/test_render_jax*.py -v

# Run with coverage
pytest src/tests/test_render_jax*.py --cov=src/render/jax --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `render.generate_jax` - Generate JAX simulation code
- `render.convert_to_jax` - Convert GNN to JAX format
- `render.optimize_jax` - Generate optimized JAX code
- `render.validate_jax` - Validate JAX model structure

### Tool Endpoints
```python
@mcp_tool("render.generate_jax")
def generate_jax_tool(model_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate JAX simulation code from GNN model"""
    return generate_jax_code(model_data, **config)
```

---

## JAX-Specific Features

### High-Performance Computing Implementation
- **JIT Compilation**: Just-in-time compilation for speed
- **Automatic Differentiation**: Reverse-mode AD for gradients
- **Vectorization**: Single-program multiple-data operations
- **GPU Acceleration**: CUDA/TPU support for acceleration
- **Memory Efficiency**: Optimized memory layouts and operations

### Optimization Strategies
- **Function Transformation**: jit, grad, vmap transformations
- **PyTree Flattening**: Efficient parameter handling
- **Random Number Generation**: Reproducible PRNG sequences
- **Memory Management**: Efficient memory allocation and reuse

---

## Development Guidelines

### Adding New JAX Features
1. Update conversion logic in `jax_renderer.py`
2. Add new JAX transformations and optimizations
3. Update template files in `templates/` directory
4. Add comprehensive performance tests

### Template Management
- Templates are stored in `templates/` directory as separate files
- Use JAX best practices for performance and memory efficiency
- Include proper JAX imports and transformations
- Maintain compatibility with different JAX versions

---

## Troubleshooting

### Common Issues

#### Issue 1: "JIT compilation failed"
**Symptom**: JAX code generation succeeds but JIT compilation fails
**Cause**: Code contains operations that cannot be JIT-compiled
**Solution**: Check for dynamic shapes or unsupported operations, use static shapes

#### Issue 2: "Gradient computation error"
**Symptom**: Automatic differentiation fails
**Cause**: Non-differentiable operations or dynamic control flow
**Solution**: Replace problematic operations with differentiable alternatives

#### Issue 3: "GPU memory error"
**Symptom**: Out of memory during GPU execution
**Cause**: Model too large for GPU memory
**Solution**: Reduce batch size, use CPU, or optimize memory usage

### Debug Mode
```python
# Enable debug output for JAX generation
result = generate_jax_code(model_data, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete JAX code generation pipeline
- JIT compilation and automatic differentiation
- GPU acceleration support
- Vectorized operations and batch processing
- Comprehensive error handling and validation
- MCP tool integration

**Known Limitations**:
- Complex control flow may impact JIT compilation
- Very large models may exceed GPU memory limits
- Some advanced JAX features require manual implementation

### Roadmap
- **Next Version**: Enhanced automatic differentiation support
- **Future**: Integration with JAX ecosystem (Flax, Optax, etc.)
- **Advanced**: Support for distributed JAX (jax.pmap)

---

## References

### Related Documentation
- [Render Module](../../render/AGENTS.md) - Parent render module
- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX docs
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) - AD theory

### External Resources
- [JIT Compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation)
- [Vector Processing](https://en.wikipedia.org/wiki/SIMD)
- [GPU Computing](https://en.wikipedia.org/wiki/GPGPU)

---

**Last Updated**: October 28, 2025
**Maintainer**: Render Module Team
**Status**: ✅ Production Ready
