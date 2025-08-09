# JAX Executor for GNN Processing Pipeline

This module provides comprehensive execution capabilities for JAX POMDP scripts generated from GNN specifications, including device selection, performance monitoring, and benchmarking.

## Features

- **Script Discovery**: Automatically finds JAX scripts in pipeline output directories
- **Device Selection**: CPU, GPU, and TPU support with automatic detection
- **Performance Monitoring**: Built-in benchmarking and resource usage tracking
- **Error Handling**: Comprehensive error reporting and recovery
- **Hardware Validation**: Automatic JAX, Optax, and Flax availability checking

## Requirements

### Core Dependencies
- **JAX**: ≥0.4.20 with appropriate hardware support
- **jaxlib**: Matching JAX version
- **Optax**: ≥0.1.7 for optimization
- **Flax**: ≥0.7.0 for neural networks
- **NumPy**: ≥1.24.0
- **SciPy**: ≥1.10.0

### Hardware Requirements
- **CPU**: Intel x86-64 (Skylake+) or AMD x86-64 (Zen2+)
- **GPU**: NVIDIA with CUDA 12.0+ (compute capability 7.0+)
- **TPU**: Google Cloud TPU v4/v5/v6

## Usage

### Command Line Interface

```bash
# Execute all JAX scripts in pipeline output
python src/execute/jax/jax_runner.py --output-dir output/ --verbose

# Execute with specific device
python src/execute/jax/jax_runner.py --output-dir output/ --device gpu

# Execute with recursive search
python src/execute/jax/jax_runner.py --output-dir output/ --recursive --verbose
```

### Python API

```python
from src.execute.jax.jax_runner import run_jax_scripts, is_jax_available

# Check JAX availability
if is_jax_available():
    # Run JAX scripts
    success = run_jax_scripts(
        pipeline_output_dir="output/",
        recursive_search=True,
        verbose=True,
        device="gpu"
    )
    print(f"Execution successful: {success}")
else:
    print("JAX not available")
```

### Individual Script Execution

```python
from src.execute.jax.jax_runner import execute_jax_script
from pathlib import Path

# Execute single JAX script
script_path = Path("output/11_render_output/jax/pomdp_solver.py")
success = execute_jax_script(script_path, verbose=True, device="cpu")
```

## Device Management

### Automatic Device Detection

The executor automatically detects available hardware:

```python
import jax
devices = jax.devices()
print(f"Available devices: {[str(d) for d in devices]}")
```

### Device Selection

```python
# CPU execution
env = {"JAX_PLATFORM_NAME": "cpu"}

# GPU execution
env = {"JAX_PLATFORM_NAME": "gpu"}

# TPU execution
env = {"JAX_PLATFORM_NAME": "tpu"}
```

## Performance Monitoring

### Built-in Benchmarking

The executor provides performance metrics:

- **Execution time** per script
- **Memory usage** tracking
- **Device utilization** monitoring
- **Error rates** and recovery

### Performance Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Performance metrics are automatically logged
# Look for messages like:
# "Script executed successfully: pomdp_solver.py (2.3s, 512MB)"
```

## Integration with Pipeline

### Step 12 Integration

The JAX executor is integrated into the main pipeline:

```python
# In 12_execute.py
from execute.jax import run_jax_scripts

def execute_jax_step(pipeline_output_dir, verbose=False):
    """Execute JAX scripts as part of pipeline step 12."""
    return run_jax_scripts(
        pipeline_output_dir=pipeline_output_dir,
        recursive_search=True,
        verbose=verbose
    )
```

### Output Directory Structure

```
output/
├── gnn_rendered_simulators/
│   └── jax/
│       ├── pomdp_solver.py
│       ├── general_model.py
│       └── combined_model.py
└── execution_results/
    └── jax/
        ├── pomdp_solver_results.json
        ├── performance_metrics.json
        └── execution_log.txt
```

## Error Handling

### Common Issues and Solutions

1. **JAX Not Available**
   ```python
   # Check installation
   is_jax_available()  # Returns False if JAX not installed
   ```

2. **Device Not Found**
   ```python
   # Use CPU fallback
   execute_jax_script(script_path, device="cpu")
   ```

3. **Memory Issues**
   ```python
   # Enable gradient checkpointing in generated code
   import jax
   jax.checkpoint = True
   ```

4. **Numerical Stability**
   ```python
   # Enable debug mode
   jax.config.update('jax_debug_nans', True)
   ```

### Error Recovery

The executor implements graceful error recovery:

- **Individual script failures** don't stop the entire pipeline
- **Device fallbacks** (GPU → CPU) when hardware unavailable
- **Retry logic** for transient failures
- **Comprehensive logging** for debugging

## Performance Optimization

### JIT Compilation

Generated JAX code uses JIT compilation for maximum performance:

```python
@partial(jit, static_argnums=(0,))
def belief_update(self, belief, action, observation):
    # JIT-compiled for performance
    pass
```

### Memory Management

- **Gradient checkpointing** for large models
- **Mixed precision** (bfloat16) for memory efficiency
- **Automatic garbage collection** after script execution

### Multi-Device Execution

```python
# Distributed execution across multiple devices
@partial(pmap, static_broadcasted_argnums=(0,))
def distributed_backup(self, belief_points, alpha_vectors):
    return self.alpha_vector_backup(belief_points, alpha_vectors)
```

## Monitoring and Debugging

### Execution Logs

```bash
# View execution logs
tail -f output/execution_results/jax/execution_log.txt
```

### Performance Metrics

```python
# Performance metrics are saved as JSON
import json
with open("output/execution_results/jax/performance_metrics.json") as f:
    metrics = json.load(f)
    print(f"Average execution time: {metrics['avg_execution_time']}s")
    print(f"Memory usage: {metrics['peak_memory_mb']}MB")
```

### Debug Mode

```python
# Enable verbose logging
logging.getLogger("src.execute.jax").setLevel(logging.DEBUG)

# Enable JAX debug features
import jax
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)
```

## Examples

### Complete Pipeline Execution

```python
from src.execute.jax.jax_runner import run_jax_scripts

# Execute all JAX scripts in pipeline
success = run_jax_scripts(
    pipeline_output_dir="output/",
    recursive_search=True,
    verbose=True,
    device="gpu"  # Use GPU if available
)

if success:
    print("All JAX scripts executed successfully")
else:
    print("Some JAX scripts failed - check logs")
```

### Custom Execution

```python
from src.execute.jax.jax_runner import find_jax_scripts, execute_jax_script

# Find specific scripts
scripts = find_jax_scripts("output/11_render_output/jax/", recursive=False)

# Execute with custom options
for script in scripts:
    if "pomdp" in script.name:
        success = execute_jax_script(script, verbose=True, device="gpu")
        print(f"{script.name}: {'Success' if success else 'Failed'}")
```

## Resources

- [JAX Documentation](https://github.com/google/jax)
- [JAX Performance Guide](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [Optax Documentation](https://optax.readthedocs.io)
- [Flax Documentation](https://flax.readthedocs.io)
- [POMDP Solvers](https://pfjax.readthedocs.io)

## Contributing

When extending the JAX executor:

1. Follow JAX best practices for performance
2. Include comprehensive error handling
3. Add performance monitoring capabilities
4. Test with various hardware configurations
5. Document new features in this README 