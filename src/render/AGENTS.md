# Render Module - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for simulation frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX)

**Pipeline Step**: Step 11: Code rendering (11_render.py)

**Category**: Code Generation / Simulation Framework Integration

**Status**: ✅ Production Ready

**Version**: 2.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. Generate simulation code for multiple frameworks
2. Convert GNN specifications to executable implementations
3. Support framework-specific optimizations
4. Provide framework compatibility validation
5. Generate documentation and usage examples

### Key Capabilities
- Multi-framework code generation (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX)
- Framework-specific optimization and configuration
- Template-based code generation with customization
- Cross-framework compatibility validation
- Automated documentation generation
- Performance optimization suggestions

### Supported Frameworks

#### PyMDP (Python)
- **Purpose**: Active Inference simulation in Python
- **Features**: Full PyMDP agent implementation
- **Output**: Complete Python simulation scripts
- **Optimization**: Matrix optimization, memory efficiency

#### RxInfer.jl (Julia)
- **Purpose**: Probabilistic programming and inference
- **Features**: Reactive inference engine
- **Output**: Julia scripts with TOML configuration
- **Optimization**: Reactive constraints, efficient inference

#### ActiveInference.jl (Julia)
- **Purpose**: Active Inference framework implementation
- **Features**: Complete Active Inference agent
- **Output**: Julia simulation scripts
- **Optimization**: Hierarchical processing, temporal dynamics

#### DisCoPy (Python)
- **Purpose**: Categorical diagrams for compositional models
- **Features**: String diagram generation
- **Output**: Python DisCoPy diagrams
- **Optimization**: Categorical composition, type checking

#### JAX (Python)
- **Purpose**: High-performance numerical computing
- **Features**: JIT compilation, automatic differentiation
- **Output**: JAX-optimized simulation code
- **Optimization**: GPU acceleration, vectorization

---

## API Reference

### Public Functions

#### `process_render(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main rendering processing function called by orchestrator (11_render.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for rendered files
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional processing options including:
  - `frameworks`: List of frameworks to render for (default: all)
  - `strict_validation`: Enable strict POMDP validation
  - `include_documentation`: Generate framework documentation

**Returns**: `True` if processing succeeded, `False` otherwise

**Example**:
```python
from render import process_render

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    verbose=True,
    frameworks=["pymdp", "rxinfer"]
)
```

#### `render_gnn_spec(gnn_spec, target, output_directory, options=None) -> Tuple[bool, str, List[str]]`
**Description**: Render a single GNN specification to a target framework

**Parameters**:
- `gnn_spec`: Parsed GNN specification dictionary
- `target`: Target framework ("pymdp", "rxinfer", "activeinference_jl", "discopy", "jax")
- `output_directory`: Output directory for generated code
- `options`: Framework-specific options

**Returns**: Tuple of (success, message, generated_files)

#### `generate_pymdp_code(model_data, output_path=None) -> str`
**Description**: Generate PyMDP simulation code

**Parameters**:
- `model_data`: GNN model data
- `output_path`: Optional output file path

**Returns**: Generated PyMDP code as string

#### `generate_rxinfer_code(model_data, output_path=None) -> str`
**Description**: Generate RxInfer.jl simulation code

**Parameters**:
- `model_data`: GNN model data
- `output_path`: Optional output file path

**Returns**: Generated RxInfer.jl code as string

#### `generate_activeinference_jl_code(model_data, output_path=None) -> str`
**Description**: Generate ActiveInference.jl simulation code

**Parameters**:
- `model_data`: GNN model data
- `output_path`: Optional output file path

**Returns**: Generated ActiveInference.jl code as string

#### `generate_discopy_code(model_data, output_path=None) -> str`
**Description**: Generate DisCoPy diagram code

**Parameters**:
- `model_data`: GNN model data
- `output_path`: Optional output file path

**Returns**: Generated DisCoPy code as string

#### `generate_jax_code(model_data, output_path=None) -> str`
**Description**: Generate JAX simulation code

**Parameters**:
- `model_data`: GNN model data
- `output_path`: Optional output file path

**Returns**: Generated JAX code as string

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `pathlib` - Path manipulation
- `typing` - Type hints

### Framework-Specific Dependencies
- **PyMDP**: `pymdp` package
- **RxInfer.jl**: Julia with RxInfer.jl package
- **ActiveInference.jl**: Julia with ActiveInference.jl package
- **DisCoPy**: `discopy` package
- **JAX**: `jax`, `jaxlib` packages

### Internal Dependencies
- `gnn.parsers` - GNN parsing and validation
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Framework Configuration
```python
PYMDP_CONFIG = {
    'inference_algorithm': 'VMP',
    'learning_rate': 0.1,
    'num_iterations': 100,
    'convergence_threshold': 1e-6
}

RXINFER_CONFIG = {
    'inference_engine': 'reactive',
    'optimization': 'auto',
    'constraints': 'default'
}
```

### Template Configuration
```python
TEMPLATE_CONFIG = {
    'include_documentation': True,
    'include_examples': True,
    'optimize_for_performance': True,
    'include_visualization': False
}
```

---

## Usage Examples

### Basic Framework Rendering
```python
from render.renderer import render_gnn_spec

# Render GNN to PyMDP
success, message, files = render_gnn_spec(
    gnn_spec=model_data,
    target="pymdp",
    output_directory="output/11_render_output",
    options={"include_examples": True}
)
```

### Multi-Framework Rendering
```python
from render.renderer import generate_pymdp_code, generate_rxinfer_code

# Generate code for multiple frameworks
pymdp_code = generate_pymdp_code(model_data)
rxinfer_code = generate_rxinfer_code(model_data)
```

### Custom Framework Options
```python
# Framework-specific options
options = {
    'pymdp': {
        'inference_algorithm': 'VMP',
        'num_iterations': 200
    },
    'rxinfer': {
        'constraints': 'custom',
        'optimization': 'performance'
    }
}
```

---

## Output Specification

### Output Products
- `*_pymdp_simulation.py` - PyMDP simulation scripts
- `*_rxinfer_simulation.jl` - RxInfer.jl simulation scripts
- `*_activeinference_simulation.jl` - ActiveInference.jl simulation scripts
- `*_discopy_diagram.py` - DisCoPy diagram scripts
- `*_jax_simulation.py` - JAX simulation scripts
- `render_processing_summary.json` - Processing summary

### Output Directory Structure
```
output/11_render_output/
├── model_name_pymdp_simulation.py
├── model_name_rxinfer_simulation.jl
├── model_name_activeinference_simulation.jl
├── model_name_discopy_diagram.py
├── model_name_jax_simulation.py
├── render_processing_summary.json
└── framework_specific_outputs/
    ├── pymdp/
    ├── rxinfer/
    └── ...
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds per framework
- **Memory**: ~50-200MB depending on model complexity
- **Status**: ✅ Production Ready

### Expected Performance
- **PyMDP Generation**: ~1-2s
- **RxInfer Generation**: ~2-4s
- **ActiveInference.jl Generation**: ~2-3s
- **DisCoPy Generation**: ~1-2s
- **JAX Generation**: ~2-4s

### Framework-Specific Performance
- **PyMDP**: Fastest, optimized Python code
- **JAX**: Fastest execution, GPU acceleration
- **RxInfer.jl**: Balanced performance and expressiveness
- **ActiveInference.jl**: Comprehensive but slower
- **DisCoPy**: Fast generation, slower execution

---

## Error Handling

### Generation Failures
1. **Syntax Errors**: Invalid GNN specification
2. **Framework Errors**: Framework-specific generation issues
3. **Dependency Errors**: Missing framework packages
4. **Configuration Errors**: Invalid framework options

### Recovery Strategies
- **Framework Fallback**: Try alternative frameworks
- **Template Simplification**: Use simpler templates
- **Partial Generation**: Generate what is possible
- **Error Documentation**: Provide detailed error reports

---

## Integration Points

### Orchestrated By
- **Script**: `11_render.py` (Step 11)
- **Function**: `process_render()`

### Imports From
- `gnn.parsers` - GNN parsing and validation
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `tests.test_render_*` - Render tests
- `execute.executor` - Execution framework integration

### Data Flow
```
GNN Parsing → Model Validation → Framework Selection → Code Generation → Framework-Specific Optimization → Output Generation
```

---

## Testing

### Test Files
- `src/tests/test_render_integration.py` - Integration tests
- `src/tests/test_render_overall.py` - Overall functionality tests
- `src/tests/test_render_performance.py` - Performance tests

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

### Key Test Scenarios
1. Multi-framework code generation
2. Framework-specific optimizations
3. Error handling and recovery
4. Performance benchmarking
5. Integration with execution step

---

## MCP Integration

### Tools Registered
- `render.generate_pymdp` - Generate PyMDP code
- `render.generate_rxinfer` - Generate RxInfer.jl code
- `render.generate_activeinference` - Generate ActiveInference.jl code
- `render.generate_discopy` - Generate DisCoPy code
- `render.generate_jax` - Generate JAX code
- `render.validate_framework` - Validate framework compatibility

### Tool Endpoints
```python
@mcp_tool("render.generate_pymdp")
def generate_pymdp_tool(model_data, options=None):
    """Generate PyMDP simulation code"""
    # Implementation
```

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready