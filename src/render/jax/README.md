# JAX Renderer for GNN Specifications

This module provides comprehensive JAX implementations for POMDPs and other Active Inference models, including optimized belief updates, value iteration, and policy optimization using JAX's advanced features.

## Features

- **JAX POMDP Solver**: Complete POMDP implementation with JIT compilation, vmap, and pmap
- **Belief Updates**: Bayesian belief updates with numerical stability
- **Value Iteration**: Alpha vector backup with vectorization
- **Performance Optimization**: JIT compilation, mixed precision, distributed computing
- **GNN Integration**: Automatic extraction of A, B, C, D matrices from GNN specifications

## Requirements

### Core Dependencies
- **JAX**: ≥0.4.20 (latest stable recommended)
- **jaxlib**: Matching JAX version
- **Optax**: ≥0.1.7 for gradient-based optimization
- **Flax**: ≥0.7.0 for neural network components
- **NumPy**: ≥1.24.0 for numerical operations
- **SciPy**: ≥1.10.0 for sparse matrix operations

### Hardware Support
- **CPU**: `pip install --upgrade jax[cpu]`
- **GPU (CUDA 12.x)**: `pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- **TPU**: `pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

## Usage

### Basic POMDP Rendering

```python
from src.render.jax import render_gnn_to_jax_pomdp

# Render GNN specification to JAX POMDP solver
success, message, files = render_gnn_to_jax_pomdp(
    gnn_spec=parsed_gnn_spec,
    output_path=Path("output/pomdp_solver.py"),
    options={"optimization_level": "high"}
)
```

### General JAX Model Rendering

```python
from src.render.jax import render_gnn_to_jax

# Render GNN specification to general JAX model
success, message, files = render_gnn_to_jax(
    gnn_spec=parsed_gnn_spec,
    output_path=Path("output/jax_model.py")
)
```

### Combined Model Rendering

```python
from src.render.jax import render_gnn_to_jax_combined

# Render GNN specification to combined JAX model
success, message, files = render_gnn_to_jax_combined(
    gnn_spec=parsed_gnn_spec,
    output_path=Path("output/combined_model.py")
)
```

## Generated Code Features

### POMDP Solver
- **JIT-compiled belief updates** for maximum performance
- **Alpha vector backup** with vectorization
- **Value iteration** with convergence checking
- **Numerical stability** in all operations
- **Device-agnostic** execution (CPU/GPU/TPU)

### Model Architecture
- **Flax-based** neural network components
- **Learnable parameters** from GNN matrices
- **Type hints** and comprehensive documentation
- **Error handling** and validation

## Performance Optimizations

### JIT Compilation
```python
@partial(jit, static_argnums=(0,))
def belief_update(self, belief, action, observation):
    # JIT-compiled belief update
    pass
```

### Vectorization
```python
# Vectorized operations across belief points
beliefs = vmap(self.belief_update)(belief_points, actions, observations)
```

### Distributed Computing
```python
# Multi-device execution
@partial(pmap, static_broadcasted_argnums=(0,))
def distributed_backup(self, belief_points, alpha_vectors):
    return self.alpha_vector_backup(belief_points, alpha_vectors)
```

## GNN Matrix Extraction

The renderer automatically extracts and parses matrices from GNN specifications:

- **A Matrix**: Observation model P(o|s)
- **B Matrix**: Transition model P(s'|s,u)
- **C Vector**: Preferences over observations
- **D Vector**: Prior over initial states

## Examples

### Simple 2-State POMDP
```python
# GNN specification with 2 states, 2 observations, 2 actions
gnn_spec = {
    "ModelName": "SimplePOMDP",
    "InitialParameterization": """
        A = [0.8, 0.2; 0.2, 0.8]
        B = [0.9, 0.1; 0.1, 0.9], [0.1, 0.9; 0.9, 0.1]
        C = [0.0, 1.0]
        D = [0.5, 0.5]
    """
}

# Generate JAX POMDP solver
render_gnn_to_jax_pomdp(gnn_spec, Path("simple_pomdp.py"))
```

### Running Generated Code
```python
# Execute generated POMDP solver
import subprocess
result = subprocess.run(["python", "simple_pomdp.py"], capture_output=True, text=True)
print(result.stdout)
```

## Integration with Pipeline

The JAX renderer is fully integrated with the GNN Processing Pipeline:

1. **Step 9 (Render)**: Generates JAX code from GNN specifications
2. **Step 10 (Execute)**: Runs generated JAX scripts with performance monitoring
3. **Setup**: Automatically installs and validates JAX dependencies

## Resources

- [JAX Documentation](https://github.com/google/jax)
- [Optax Documentation](https://optax.readthedocs.io)
- [Flax Documentation](https://flax.readthedocs.io)
- [PFJAX Documentation](https://pfjax.readthedocs.io)
- [POMDP Theory](https://arxiv.org/abs/1304.1118)
- [Point-Based Value Iteration](https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf)

## Troubleshooting

### Common Issues

1. **JAX Installation**: Ensure correct version for your hardware
2. **Memory Issues**: Use gradient checkpointing for large models
3. **Performance**: Enable JIT compilation and use appropriate devices
4. **Numerical Stability**: Check for NaN values in belief updates

### Debug Mode
```python
import jax
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)
```

## Contributing

When extending the JAX renderer:

1. Follow JAX best practices for performance
2. Include comprehensive docstrings with @Web links
3. Add type hints and error handling
4. Test with various GNN specifications
5. Document new features in this README
