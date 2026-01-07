# Render Module - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for simulation frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX)

**Pipeline Step**: Step 11: Code rendering (11_render.py)

**Category**: Code Generation / Simulation Framework Integration

**Status**: ✅ Production Ready

**Version**: 2.0.0

**Last Updated**: 2026-01-07

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

#### `process_render(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main rendering processing function called by orchestrator (11_render.py). Processes GNN files and generates code for multiple simulation frameworks.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for rendered files
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional processing options including:
  - `frameworks` (List[str]): List of frameworks to render for (default: ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy"])
  - `strict_validation` (bool): Enable strict POMDP validation (default: False)
  - `include_documentation` (bool): Generate framework documentation (default: True)
  - `pomdp_aware` (bool): Enable POMDP-aware processing (default: True)

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from render import process_render
from pathlib import Path

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    verbose=True,
    frameworks=["pymdp", "rxinfer"],
    strict_validation=True
)
```

#### `render_gnn_spec(gnn_spec: Dict[str, Any], target: str, output_directory: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
**Description**: Render a GNN specification dictionary to a target framework.

**Parameters**:
- `gnn_spec` (Dict[str, Any]): Parsed GNN specification dictionary
- `target` (str): Target framework ("pymdp", "rxinfer", "activeinference_jl", "jax", "discopy")
- `output_directory` (Union[str, Path]): Output directory for generated code
- `options` (Optional[Dict[str, Any]]): Framework-specific options (default: None)

**Returns**: `Tuple[bool, str, List[str]]` - Tuple containing:
- `success` (bool): Whether rendering succeeded
- `message` (str): Status message
- `generated_files` (List[str]): List of generated file paths

**Location**: `src/render/processor.py`

#### `generate_pymdp_code(gnn_content: str, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Generate PyMDP simulation code from GNN content.

**Parameters**:
- `gnn_content` (str): GNN specification content
- `output_dir` (Path): Output directory for PyMDP code
- `**kwargs`: Additional PyMDP-specific options

**Returns**: `Dict[str, Any]` - Generation results with:
- `success` (bool): Whether generation succeeded
- `main_script` (str): Path to main PyMDP script
- `files` (List[str]): List of generated files
- `config` (Dict[str, Any]): PyMDP configuration

#### `generate_rxinfer_code(gnn_content: str, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Generate RxInfer.jl simulation code from GNN content.

**Parameters**:
- `gnn_content` (str): GNN specification content
- `output_dir` (Path): Output directory for RxInfer.jl code
- `**kwargs`: Additional RxInfer.jl-specific options

**Returns**: `Dict[str, Any]` - Generation results with:
- `success` (bool): Whether generation succeeded
- `main_script` (str): Path to main Julia script
- `toml_config` (str): Path to TOML configuration file
- `files` (List[str]): List of generated files

#### `generate_activeinference_jl_code(gnn_content: str, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Generate ActiveInference.jl simulation code from GNN content.

**Parameters**:
- `gnn_content` (str): GNN specification content
- `output_dir` (Path): Output directory for ActiveInference.jl code
- `**kwargs`: Additional ActiveInference.jl-specific options

**Returns**: `Dict[str, Any]` - Generation results with:
- `success` (bool): Whether generation succeeded
- `main_script` (str): Path to main Julia script
- `files` (List[str]): List of generated files

#### `generate_discopy_code(gnn_content: str, output_dir: Path, **kwargs) -> Dict[str, Any]`
**Description**: Generate DisCoPy diagram code from GNN content.

**Parameters**:
- `gnn_content` (str): GNN specification content
- `output_dir` (Path): Output directory for DisCoPy code
- `**kwargs`: Additional DisCoPy-specific options

**Returns**: `Dict[str, Any]` - Generation results with:
- `success` (bool): Whether generation succeeded
- `diagram_script` (str): Path to diagram script
- `files` (List[str]): List of generated files

#### `render_gnn_spec(gnn_spec: Dict[str, Any], target: str, output_directory: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`
**Description**: Render a GNN specification dictionary to a target framework.

**Parameters**:
- `gnn_spec` (Dict[str, Any]): Parsed GNN specification dictionary
- `target` (str): Target framework ("pymdp", "rxinfer", "activeinference_jl", "discopy", "jax")
- `output_directory` (Union[str, Path]): Output directory for generated code
- `options` (Optional[Dict[str, Any]]): Framework-specific options (default: None)

**Returns**: `Tuple[bool, str, List[str]]` - Tuple containing:
- `success` (bool): Whether rendering succeeded
- `message` (str): Status message
- `generated_files` (List[str]): List of generated file paths

**Location**: `src/render/processor.py`

#### `get_module_info() -> Dict[str, Any]`
**Description**: Get information about the enhanced render module.

**Returns**: `Dict[str, Any]` - Dictionary with module information containing:
- `name` (str): Module name
- `version` (str): Module version
- `description` (str): Module description
- `supported_targets` (List[str]): List of supported target frameworks
- `available_targets` (List[str]): List of currently available targets
- `features` (List[str]): List of available features
- `supported_formats` (List[str]): List of supported output formats
- `processing_modes` (List[str]): List of available processing modes

**Location**: `src/render/processor.py`

#### `get_available_renderers() -> Dict[str, Dict[str, Any]]`
**Description**: Get information about available renderers for each framework.

**Returns**: `Dict[str, Dict[str, Any]]` - Dictionary mapping framework names to renderer information:
- Each framework entry contains:
  - `name` (str): Framework name
  - `description` (str): Framework description
  - `language` (str): Target language
  - `file_extension` (str): Output file extension
  - `supported_features` (List[str]): List of supported features
  - `function` (str): Function name for rendering
  - `output_format` (str): Output format type
  - `pomdp_compatible` (bool): Whether POMDP-aware processing is supported

**Location**: `src/render/processor.py`

#### `validate_pomdp_for_rendering(pomdp_space: Any) -> Tuple[bool, List[str]]`
**Description**: Validate POMDP state space structure for rendering compatibility.

**Parameters**:
- `pomdp_space` (Any): POMDP state space object to validate

**Returns**: `Tuple[bool, List[str]]` - Tuple containing:
- `is_valid` (bool): Whether POMDP structure is valid
- `errors` (List[str]): List of validation error messages

**Location**: `src/render/processor.py`

#### `normalize_matrices(pomdp_space: Any, logger) -> Any`
**Description**: Normalize POMDP matrices for consistent rendering.

**Parameters**:
- `pomdp_space` (Any): POMDP state space object
- `logger`: Logger instance for logging

**Returns**: `Any` - Normalized POMDP state space object

**Location**: `src/render/processor.py`

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

### MCP File Location
- `src/render/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Framework-specific rendering fails
**Symptom**: Code generation fails for specific framework  
**Cause**: Missing framework dependencies or invalid GNN model structure  
**Solution**: 
- Check framework dependencies are installed
- Verify GNN model has required sections for framework
- Use `--verbose` flag for detailed error messages
- Check framework-specific requirements in documentation

#### Issue 2: POMDP validation errors
**Symptom**: POMDP-aware rendering reports validation errors  
**Cause**: GNN model missing POMDP-required components or invalid structure  
**Solution**:
- Ensure GNN model has complete state space, observations, actions
- Verify connections follow POMDP structure (s->o, s->s, a->s)
- Use `--strict-validation=False` for lenient validation
- Review POMDP requirements in documentation

#### Issue 3: Generated code doesn't execute
**Symptom**: Rendered code has syntax errors or import failures  
**Cause**: Framework version mismatch or template issues  
**Solution**:
- Verify framework versions match requirements
- Check generated code for syntax errors
- Review framework-specific documentation
- Report template issues if systematic

---

## Version History

### Current Version: 2.0.0

**Features**:
- Multi-framework code generation (PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX)
- POMDP-aware processing
- Framework-specific optimization
- Template-based generation
- Cross-framework compatibility validation

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Additional framework support
- **Future**: Real-time code generation API

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [PyMDP Integration](../../doc/pymdp/)
- [RxInfer Integration](../../doc/rxinfer/)
- [ActiveInference.jl Integration](../../doc/activeinference_jl/)
- [DisCoPy Integration](../../doc/discopy/)

### External Resources
- [PyMDP Framework](https://github.com/infer-actively/pymdp)
- [RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- [DisCoPy](https://github.com/oxford-quantum-group/discopy)
- [JAX Documentation](https://jax.readthedocs.io/)

---

**Last Updated**: 2026-01-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 2.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern