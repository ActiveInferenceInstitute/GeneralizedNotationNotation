# JAX Execute Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Execution and performance monitoring of JAX-based simulations generated from GNN specifications

**Parent Module**: Execute Module (Step 12: Simulation execution)

**Category**: Framework Execution / JAX Performance

---

## Core Functionality

### Primary Responsibilities
1. Execute JAX simulation scripts with optimized performance
2. Manage device selection (CPU, GPU, TPU) with automatic detection
3. Provide comprehensive performance monitoring and benchmarking
4. Handle JAX-specific optimizations and memory management
5. Ensure hardware compatibility and graceful degradation

### Key Capabilities
- JIT-compiled simulation execution with hardware acceleration
- Automatic device detection and selection
- Performance profiling and resource monitoring
- Memory-efficient execution with gradient checkpointing
- Error recovery and fallback strategies
- Hardware validation and compatibility checking

---

## API Reference

### Public Functions

#### `run_jax_scripts(pipeline_output_dir: Union[str, Path], recursive_search: bool = True, **kwargs) -> bool`
**Description**: Execute all JAX scripts found in pipeline output directory.

**Parameters**:
- `pipeline_output_dir` (Union[str, Path]): Directory containing pipeline outputs
- `recursive_search` (bool): Whether to search subdirectories (default: True)
- `device` (str, optional): Device to use ("auto", "cpu", "gpu", "tpu") (default: "auto")
- `timeout` (int, optional): Execution timeout per script in seconds (default: 300)
- `verbose` (bool, optional): Enable verbose logging (default: False)
- `**kwargs`: Additional execution options

**Returns**: `bool` - True if all scripts executed successfully, False otherwise

**Example**:
```python
from execute.jax import run_jax_scripts
from pathlib import Path

success = run_jax_scripts(
    pipeline_output_dir=Path("output/11_render_output"),
    recursive_search=True,
    device="gpu",
    verbose=True,
    timeout=600
)
```

#### `execute_jax_script(script_path: Union[str, Path], device: str = "auto", **kwargs) -> Dict[str, Any]`
**Description**: Execute a specific JAX script with performance monitoring and hardware acceleration.

**Parameters**:
- `script_path` (Union[str, Path]): Path to JAX script file
- `device` (str, optional): Device to use ("auto", "cpu", "gpu", "tpu") (default: "auto")
- `profile` (bool, optional): Enable performance profiling (default: False)
- `timeout` (int, optional): Execution timeout in seconds (default: 300)
- `**kwargs`: Additional execution options

**Returns**: `Dict[str, Any]` - Execution results dictionary with:
- `success` (bool): Whether execution succeeded
- `execution_time` (float): Execution time in seconds
- `device_used` (str): Device that was used
- `performance_metrics` (Dict): Performance metrics if profiling enabled
- `output_files` (List[Path]): Generated output files

#### `find_jax_scripts(search_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
**Description**: Find all JAX script files in a directory.

**Parameters**:
- `search_dir` (Union[str, Path]): Directory to search in
- `recursive` (bool): Whether to search subdirectories (default: True)

**Returns**: `List[Path]` - List of paths to JAX script files

#### `is_jax_available(device: str = "cpu") -> bool`
**Description**: Check if JAX is available and configured for specified device.

**Parameters**:
- `device` (str): Device to check availability for

**Returns**: `True` if JAX is available for the device

#### `get_jax_device_info() -> Dict[str, Any]`
**Description**: Get information about available JAX devices and capabilities

**Returns**: Dictionary with device information and capabilities

---

## Dependencies

### Required Dependencies
- `jax` - JAX library for high-performance computing
- `jaxlib` - JAX compilation backend with hardware support
- `numpy` - Array operations and numerical computing

### Optional Dependencies
- `optax` - Gradient-based optimization (fallback: basic SGD)
- `flax` - Neural network library (fallback: basic operations)
- `psutil` - System resource monitoring (fallback: basic monitoring)

### Internal Dependencies
- `execute.executor` - Base execution functionality
- `render.jax` - JAX code generation
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### JAX Execution Configuration
```python
JAX_EXEC_CONFIG = {
    'device_selection': 'auto',              # Device selection strategy
    'memory_limit': 'auto',                  # Memory limit per script
    'timeout_seconds': 300,                  # Execution timeout
    'enable_profiling': True,                # Enable performance profiling
    'checkpoint_gradients': True,            # Enable gradient checkpointing
    'mixed_precision': True,                 # Enable mixed precision
    'parallel_execution': False,             # Enable parallel script execution
    'cleanup_temp_files': True               # Clean temporary files
}
```

### Device Configuration
```python
DEVICE_CONFIG = {
    'cpu': {
        'platform_name': 'cpu',
        'thread_count': None,                 # Auto-detect
        'memory_limit': '8GB'
    },
    'gpu': {
        'platform_name': 'gpu',
        'device_id': 0,                       # Primary GPU
        'memory_limit': 'auto',               # Use available memory
        'enable_peer_access': False
    },
    'tpu': {
        'platform_name': 'tpu',
        'memory_limit': '32GB',
        'core_count': 8
    }
}
```

### Performance Monitoring Configuration
```python
MONITORING_CONFIG = {
    'metrics': {
        'execution_time': True,
        'memory_usage': True,
        'cpu_utilization': True,
        'gpu_utilization': True,
        'jit_compile_time': True
    },
    'logging': {
        'level': 'INFO',
        'file_output': True,
        'console_output': True,
        'performance_summary': True
    },
    'benchmarking': {
        'warmup_runs': 1,
        'measurement_runs': 3,
        'statistical_analysis': True
    }
}
```

---

## Usage Examples

### Basic JAX Script Execution
```python
from execute.jax import run_jax_scripts

# Execute all JAX scripts in pipeline output
success = run_jax_scripts(
    pipeline_output_dir="output/11_render_output",
    recursive_search=True,
    device="auto",  # Auto-select best available device
    verbose=True
)

if success:
    print("All JAX scripts executed successfully")
else:
    print("Some JAX scripts failed - check execution logs")
```

### Execute Specific Script with Monitoring
```python
from execute.jax import execute_jax_script

# Execute specific JAX script with detailed monitoring
script_path = "output/11_render_output/jax/pomdp_solver.py"
results = execute_jax_script(
    script_path=script_path,
    device="gpu",
    enable_profiling=True,
    timeout_seconds=600
)

print(f"Execution time: {results['execution_time']:.2f}s")
print(f"Peak memory: {results['peak_memory_mb']:.1f}MB")
print(f"Success: {results['success']}")
```

### Device Detection and Selection
```python
from execute.jax import get_jax_device_info, is_jax_available

# Get available device information
device_info = get_jax_device_info()
print("Available devices:")
for device in device_info['available_devices']:
    print(f"  - {device['name']}: {device['memory_gb']:.1f}GB")

# Check specific device availability
gpu_available = is_jax_available("gpu")
tpu_available = is_jax_available("tpu")

print(f"GPU available: {gpu_available}")
print(f"TPU available: {tpu_available}")
```

### Performance Benchmarking
```python
from execute.jax import execute_jax_script

# Execute with detailed performance benchmarking
results = execute_jax_script(
    script_path="jax_simulation.py",
    device="gpu",
    enable_profiling=True,
    benchmark_runs=5
)

print("Performance Results:")
print(f"Mean execution time: {results['benchmark']['mean_time']:.3f}s")
print(f"Std deviation: {results['benchmark']['std_time']:.3f}s")
print(f"Memory efficiency: {results['benchmark']['memory_efficiency']:.2f}")
```

### Script Discovery and Batch Execution
```python
from execute.jax import find_jax_scripts, execute_jax_script

# Find all JAX scripts in directory
scripts = find_jax_scripts("output/11_render_output/jax/", recursive=True)
print(f"Found {len(scripts)} JAX scripts")

# Execute scripts in parallel (if supported)
for script_path in scripts:
    results = execute_jax_script(script_path, device="auto", verbose=False)
    status = "✓" if results['success'] else "✗"
    print(f"{status} {script_path}: {results['execution_time']:.2f}s")
```

---

## Device Management and Hardware Acceleration

### Automatic Device Selection
- **Hardware Detection**: Automatically detect available CPU, GPU, and TPU devices
- **Performance Ranking**: Rank devices by expected performance
- **Fallback Strategy**: Graceful fallback to slower devices when preferred hardware unavailable
- **Memory Management**: Automatic memory allocation based on device capabilities

### GPU Acceleration Features
- **CUDA Support**: Full CUDA compatibility with optimized kernels
- **Memory Pooling**: Efficient GPU memory management
- **Multi-GPU Support**: Distributed execution across multiple GPUs
- **Peer Access**: Direct GPU-to-GPU communication when available

### TPU Acceleration Features
- **TPU v4/v5 Support**: Latest TPU generation compatibility
- **XLA Compilation**: Automatic XLA optimization for TPU execution
- **Pod Support**: Multi-TPU pod execution for large models
- **BFloat16 Support**: Mixed precision training and inference

---

## Performance Monitoring and Profiling

### Built-in Metrics Collection
- **Execution Time**: Wall-clock time and CPU time tracking
- **Memory Usage**: Peak memory usage and memory efficiency
- **Device Utilization**: GPU/TPU utilization percentages
- **JIT Compilation**: Compilation time and optimization effectiveness

### Profiling Capabilities
- **Function-level Profiling**: Profile individual JAX functions
- **Memory Tracing**: Track memory allocation and deallocation
- **Gradient Computation**: Profile automatic differentiation performance
- **Kernel Execution**: Monitor low-level kernel performance

### Benchmarking Suite
- **Warm-up Runs**: Initial runs to stabilize performance
- **Statistical Analysis**: Mean, variance, and confidence intervals
- **Comparative Benchmarking**: Compare performance across devices
- **Regression Detection**: Detect performance regressions over time

---

## Output Specification

### Output Products
- `jax_execution_results.json` - Complete execution results summary
- `jax_performance_metrics.json` - Detailed performance metrics
- `jax_execution_log.txt` - Execution log with timestamps
- `jax_device_info.json` - Hardware and device information
- `jax_benchmark_results.json` - Benchmarking results (if enabled)

### Output Directory Structure
```
output/12_execute_output/
├── jax_results/
│   ├── jax_execution_results.json
│   ├── jax_performance_metrics.json
│   ├── jax_execution_log.txt
│   ├── jax_device_info.json
│   └── script_results/
│       ├── pomdp_solver_result.json
│       ├── general_model_result.json
│       └── combined_model_result.json
└── jax_profiling/
    ├── memory_profile.json
    └── timing_profile.json
```

### Results Data Structure
```python
execution_results = {
    'metadata': {
        'framework': 'jax',
        'device_used': 'gpu',
        'jax_version': '0.4.20',
        'execution_timestamp': '2025-10-28T10:30:00Z',
        'scripts_executed': 3,
        'scripts_successful': 3
    },
    'device_info': {
        'name': 'NVIDIA RTX 3080',
        'memory_gb': 10.0,
        'compute_capability': '8.6',
        'driver_version': '525.60.13'
    },
    'performance_summary': {
        'total_execution_time': 45.67,
        'average_memory_usage': 2.3,
        'peak_memory_usage': 4.1,
        'device_utilization': 0.87
    },
    'script_results': [
        {
            'script_name': 'pomdp_solver.py',
            'success': True,
            'execution_time': 12.34,
            'memory_peak': 2.1,
            'jit_compile_time': 2.56,
            'results': {...}  # Script-specific results
        }
    ],
    'benchmarking': {
        'runs_completed': 5,
        'mean_execution_time': 11.98,
        'std_execution_time': 0.45,
        'performance_stability': 0.96
    }
}
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 5-120 seconds per script batch
- **Memory**: 200MB-4GB depending on script complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Device Detection**: < 1s
- **Script Discovery**: < 2s
- **JIT Compilation**: 2-10s (first run, cached thereafter)
- **Execution**: 1-100s (main computation time)
- **Result Processing**: 1-5s
- **Cleanup**: < 1s

### Optimization Notes
- JAX JIT compilation provides significant speedups on repeated executions
- GPU acceleration can provide 10-100x speedup over CPU for compatible operations
- Memory usage scales with model size and batch processing requirements
- Mixed precision can reduce memory usage by 50% with minimal accuracy loss

---

## Error Handling

### JAX Execution Errors
1. **Device Not Available**: Requested device not found or not accessible
2. **Memory Allocation Failed**: Insufficient memory for model execution
3. **JIT Compilation Failed**: Code cannot be JIT-compiled
4. **Numerical Instability**: NaN or Inf values during computation
5. **Timeout Exceeded**: Script execution takes too long

### Recovery Strategies
- **Device Fallback**: Automatic fallback to available devices (GPU → CPU)
- **Memory Optimization**: Gradient checkpointing and memory pooling
- **Code Restructuring**: Automatic code modifications for JIT compatibility
- **Numerical Stabilization**: Automatic detection and correction of instabilities

### Error Examples
```python
try:
    results = execute_jax_script(script_path, device="gpu")
except JAXExecutionError as e:
    logger.error(f"JAX execution failed: {e}")
    # Attempt recovery with CPU fallback
    results = execute_jax_script(script_path, device="cpu")
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/execute/` (Step 12)
- **Main Script**: `12_execute.py`

### Imports From
- `render.jax` - JAX code generation
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `execute.processor` - Main execution integration
- `tests.test_execute_jax*` - JAX-specific tests

### Data Flow
```
JAX Code Generation → Device Selection → Hardware Validation → JIT Compilation → Script Execution → Performance Monitoring → Result Aggregation
```

---

## Testing

### Test Files
- `src/tests/test_execute_jax_integration.py` - Integration tests
- `src/tests/test_execute_jax_performance.py` - Performance tests
- `src/tests/test_execute_jax_devices.py` - Device management tests

### Test Coverage
- **Current**: 82%
- **Target**: 90%+

### Key Test Scenarios
1. Device detection and selection across different hardware
2. Script execution with various JAX features (JIT, vmap, grad)
3. Performance monitoring and benchmarking accuracy
4. Error handling and recovery mechanisms
5. Memory management and optimization

### Test Commands
```bash
# Run JAX execution tests
pytest src/tests/test_execute_jax*.py -v

# Run with coverage
pytest src/tests/test_execute_jax*.py --cov=src/execute/jax --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `execute.run_jax_scripts` - Execute JAX simulation scripts
- `execute.check_jax_availability` - Check JAX environment availability
- `execute.monitor_jax_performance` - Monitor JAX execution performance
- `execute.benchmark_jax_script` - Benchmark JAX script performance

### Tool Endpoints
```python
@mcp_tool("execute.run_jax_scripts")
def run_jax_scripts_tool(pipeline_output_dir: str, device: str = "auto") -> Dict[str, Any]:
    """Execute JAX simulation scripts with performance monitoring"""
    return run_jax_scripts(pipeline_output_dir, device=device, return_detailed_results=True)
```

---

## JAX-Specific Optimizations

### JIT Compilation Features
- **Function Specialization**: Compile functions for specific argument shapes
- **Kernel Fusion**: Automatically fuse operations for efficiency
- **Memory Layout Optimization**: Optimize array memory layouts
- **Loop Unrolling**: Automatic loop unrolling for small iterations

### Memory Management
- **Gradient Checkpointing**: Trade computation for memory in deep networks
- **Memory Pooling**: Reuse allocated memory across operations
- **Automatic Garbage Collection**: Efficient memory cleanup
- **Memory Defragmentation**: Optimize memory layout during execution

### Hardware-Specific Optimizations
- **GPU Kernel Optimization**: Custom kernels for common operations
- **TPU Pod Scaling**: Efficient scaling across TPU cores
- **CPU Vectorization**: SIMD instruction utilization
- **Multi-Device Parallelism**: Distributed execution patterns

---

## Development Guidelines

### Adding New JAX Features
1. Update execution logic in `jax_runner.py`
2. Add new device support and optimizations
3. Update performance monitoring capabilities
4. Add comprehensive tests for new features

### Performance Optimization
- Profile JAX operations to identify bottlenecks
- Use appropriate JAX transformations (jit, vmap, grad)
- Optimize memory usage and data transfer
- Consider hardware-specific optimizations

---

## Troubleshooting

### Common Issues

#### Issue 1: "JAX device not available"
**Symptom**: Execution fails with device access error
**Cause**: Requested device not installed or not accessible
**Solution**: Check device availability and use fallback device

#### Issue 2: "JIT compilation failed"
**Symptom**: Script fails during JIT compilation phase
**Cause**: Code contains non-JIT-compatible operations
**Solution**: Restructure code to use JIT-compatible operations or disable JIT

#### Issue 3: "Out of memory error"
**Symptom**: Execution fails with memory allocation error
**Cause**: Model too large for available memory
**Solution**: Enable gradient checkpointing, use smaller batches, or switch to CPU

### Debug Mode
```python
# Enable debug output for JAX execution
results = execute_jax_script(
    script_path,
    device="gpu",
    debug=True,
    verbose=True,
    enable_jax_debug=True
)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete JAX execution pipeline with hardware acceleration
- Automatic device detection and selection
- Comprehensive performance monitoring and benchmarking
- Memory optimization with gradient checkpointing
- Extensive error handling and recovery
- MCP tool integration

**Known Limitations**:
- Some advanced JAX features require manual configuration
- TPU execution requires Google Cloud environment
- Memory usage can be high for large models

### Roadmap
- **Next Version**: Enhanced multi-device execution support
- **Future**: Automatic performance optimization
- **Advanced**: Integration with JAX ecosystem (Jaxopt, Jraph, etc.)

---

## References

### Related Documentation
- [Execute Module](../../execute/AGENTS.md) - Parent execute module
- [JAX Render](../../render/jax/AGENTS.md) - JAX code generation
- [JAX Documentation](https://jax.readthedocs.io/) - Official JAX docs

### External Resources
- [JAX GitHub](https://github.com/google/jax)
- [JAX Performance Guide](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [CUDA Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

**Last Updated**: 2025-12-30
**Maintainer**: Execute Module Team
**Status**: ✅ Production Ready




