# Render (Code Generation) - Agent Scaffolding

## Module Overview

**Purpose**: Generate executable code for multiple Active Inference simulation frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy) from parsed GNN specifications

**Pipeline Step**: Step 11: Code rendering (11_render.py)

**Category**: Code Generation / Translation

---

## Core Functionality

### Primary Responsibilities
1. Extract POMDP specifications from parsed GNN models
2. Generate framework-specific executable code
3. Create comprehensive documentation for each rendering
4. Validate generated code syntax

### Key Capabilities
- **PyMDP**: Python simulation code for discrete active inference
- **RxInfer.jl**: Julia reactive message-passing code
- **ActiveInference.jl**: Julia generative model code  
- **JAX**: Python/JAX code for GPU-accelerated inference
- **DisCoPy**: Categorical diagram specifications

---

## API Reference

### Public Functions

#### `process_render(target_dir, output_dir, **kwargs) -> bool`
**Description**: Main rendering function that processes GNN files and generates code for all frameworks

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for generated code
- `**kwargs`: Additional options (render_format, target_language, verbose)

**Returns**: `True` if rendering succeeded, `False` otherwise

**Example**:
```python
from render import process_render

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    render_format="all",
    verbose=True
)
```

#### `render_gnn_spec(gnn_spec, framework, output_dir) -> Path`
**Description**: Render a single GNN specification to a specific framework

**Parameters**:
- `gnn_spec` (dict): Parsed GNN specification
- `framework` (str): Target framework ("pymdp", "rxinfer", etc.)
- `output_dir` (Path): Output directory

**Returns**: Path to generated code file

---

### Public Classes

#### `POMDPProcessor`
**Description**: Extracts and processes POMDP specifications from GNN models

**Methods**:
- `extract_pomdp(gnn_model) -> POMDP` - Extract POMDP from model
- `validate_pomdp(pomdp) -> bool` - Validate POMDP structure
- `get_dimensions(pomdp) -> Dict` - Get state/obs/action dimensions

**Example**:
```python
processor = POMDPProcessor()
pomdp = processor.extract_pomdp(gnn_model)
dims = processor.get_dimensions(pomdp)
```

#### `FrameworkRenderer` (Base Class)
**Description**: Base class for framework-specific renderers

**Subclasses**:
- `PyMDPRenderer` - PyMDP code generation
- `RxInferRenderer` - RxInfer.jl code generation
- `ActiveInferenceRenderer` - ActiveInference.jl code generation
- `JAXRenderer` - JAX code generation
- `DisCoPyRenderer` - DisCoPy diagram generation

---

## Dependencies

### Required Dependencies
- `pathlib` - File path manipulation
- `jinja2` - Template engine for code generation
- `json` - Configuration and metadata

### Optional Dependencies
- `pymdp` - For validation of generated PyMDP code (fallback: skip validation)
- `julia` - For Julia code syntax checking (fallback: skip validation)

### Internal Dependencies
- `utils.pipeline_template` - Logging and utilities
- `pipeline.config` - Configuration management
- `gnn.multi_format_processor` - GNN model loading

---

## Configuration

### Environment Variables
- `JULIA_PATH` - Path to Julia executable (default: "julia")
- `RENDER_VALIDATE_OUTPUT` - Enable output validation (default: False)

### Default Settings
```python
DEFAULT_FRAMEWORKS = ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy"]
DEFAULT_TEMPLATE_DIR = "src/render/templates"
VALIDATION_ENABLED = False
```

---

## Usage Examples

### Basic Usage
```python
from render import process_render

success = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output")
)
```

### Framework-Specific Rendering
```python
from render import render_gnn_spec
from gnn import load_parsed_model

model = load_parsed_model("actinf_pomdp_agent.md")
pymdp_file = render_gnn_spec(model, "pymdp", Path("output"))
```

### Pipeline Integration
```python
# Called from 11_render.py
from render import process_render

run_script = create_standardized_pipeline_script(
    "11_render.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_render_processing(
        target_dir, output_dir, logger, **kwargs
    ),
    "Render processing for GNN specifications"
)
```

---

## Input/Output Specification

### Input Requirements
- **File Formats**: Parsed GNN JSON files from step 3
- **Directory Structure**: `output/3_gnn_output/model_name/model_name_parsed.json`
- **Prerequisites**: Step 3 (GNN file processing) must complete first

### Output Products
- **Primary Outputs**: 
  - `model_name_pymdp.py` - PyMDP simulation code
  - `model_name_rxinfer.jl` - RxInfer.jl code
  - `model_name_activeinference.jl` - ActiveInference.jl code
  - `model_name_jax.py` - JAX code
  - `model_name_discopy.py` - DisCoPy diagram code
- **Metadata Files**: 
  - `render_processing_summary.json` - Rendering summary
  - `README.md` - Overview documentation
- **Artifacts**: Framework-specific documentation

### Output Directory Structure
```
output/11_render_output/
├── model_name/
│   ├── pymdp/
│   │   ├── model_name_pymdp.py
│   │   └── README.md
│   ├── rxinfer/
│   │   ├── model_name_rxinfer.jl
│   │   └── README.md
│   ├── activeinference_jl/
│   │   ├── model_name_activeinference.jl
│   │   └── README.md
│   ├── jax/
│   │   ├── model_name_jax.py
│   │   └── README.md
│   └── discopy/
│       ├── model_name_discopy.py
│       └── README.md
├── render_processing_summary.json
└── README.md
```

---

## Error Handling

### Error Categories
1. **POMDP Extraction Errors**: Invalid model structure
2. **Template Errors**: Missing or invalid templates
3. **Validation Errors**: Generated code syntax errors
4. **Framework Errors**: Framework-specific issues

### Fallback Strategies
- **Primary**: Generate code for all 5 frameworks
- **Fallback 1**: Skip problematic framework, continue with others
- **Fallback 2**: Generate minimal template-based code
- **Final**: Create placeholder with error documentation

### Error Reporting
- **Logging Level**: ERROR for critical failures, WARNING for framework skips
- **User Messages**: "Failed to render {framework}: {specific_error}"
- **Recovery Suggestions**: "Check POMDP extraction" or "Install {framework} for validation"

---

## Integration Points

### Orchestrated By
- **Script**: `11_render.py`
- **Function**: `_run_render_processing()`

### Imports From
- `utils.pipeline_template` - Logging utilities
- `pipeline.config` - Output directory management
- `gnn.multi_format_processor` - Model loading

### Imported By
- `12_execute.py` - Executes generated code
- `tests.test_render_integration.py` - Integration tests

### Data Flow
```
output/3_gnn_output/parsed.json → Render Module → output/11_render_output/framework_code
                                        ↓
                              [Step 12: Execute]
```

---

## Testing

### Test Files
- `src/tests/test_render_integration.py` - Integration tests
- `src/tests/test_render_pymdp.py` - PyMDP renderer tests
- `src/tests/test_render_julia.py` - Julia renderer tests
- `src/tests/test_render_jax.py` - JAX renderer tests

### Test Coverage
- **Current**: 78%
- **Target**: 90%+

### Key Test Scenarios
1. Extract valid POMDP from GNN model
2. Generate syntactically correct code for each framework
3. Handle missing optional dependencies gracefully
4. Validate generated code structure
5. Create comprehensive documentation

### Test Commands
```bash
# Run render-specific tests
pytest src/tests/test_render*.py -v

# Run with coverage
pytest src/tests/test_render*.py --cov=src/render --cov-report=term-missing

# Test specific framework
pytest src/tests/test_render_pymdp.py -v
```

---

## MCP Integration

### Tools Registered
- `render_model` - Render GNN model to framework
- `list_renderers` - List available rendering frameworks
- `validate_code` - Validate generated code

### Tool Endpoints
```python
@mcp_tool("render_model")
def render_tool(model_path: str, framework: str):
    """Render a GNN model to specified framework"""
    return render_gnn_spec(model_path, framework, output_dir)
```

### MCP File Location
- `src/render/mcp.py` - MCP tool registrations

---

## Performance Characteristics

### Resource Requirements
- **Memory**: ~10MB per model (template rendering)
- **CPU**: Low (primarily template processing)
- **Disk**: ~50KB per generated file × 5 frameworks = ~250KB per model

### Execution Time
- **Fast Path**: ~100ms for typical model (all frameworks)
- **Slow Path**: ~500ms for complex models (>50 variables)
- **Timeout**: None (synchronous processing)

### Scalability
- **Input Size Limits**: No inherent limits (limited by GNN parser)
- **Parallelization**: Could parallelize framework rendering

---

## Development Guidelines

### Adding New Frameworks
1. Create renderer class in `src/render/[framework]/`
2. Implement `render()` method
3. Add templates to `src/render/templates/[framework]/`
4. Add tests for new framework
5. Update documentation

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all public functions
- Document all renderer classes
- Include usage examples in docstrings

### Testing Requirements
- All new renderers must have integration tests
- Generated code must be syntactically valid
- Coverage must remain >75%

---

## Troubleshooting

### Common Issues

#### Issue 1: "POMDP extraction failed"
**Symptom**: Render step fails to extract POMDP  
**Cause**: Invalid or incomplete GNN model structure  
**Solution**: Check GNN model has required POMDP components (states, observations, actions)

#### Issue 2: "Template not found for framework {X}"
**Symptom**: Missing template error  
**Cause**: Template file deleted or moved  
**Solution**: Restore template from repository or create custom template

#### Issue 3: "Generated code syntax invalid"
**Symptom**: Validation warnings for generated code  
**Cause**: Complex model structure or template bug  
**Solution**: Report issue with model details, use generated code as starting point

### Debug Mode
```bash
# Run with verbose logging
python src/11_render.py --verbose

# Check output directory
ls -la output/11_render_output/

# View rendering summary
cat output/11_render_output/render_processing_summary.json | python -m json.tool

# Test generated code
python output/11_render_output/model/pymdp/model_pymdp.py
```

---

## Version History

### Current Version: 2.1.0

**Features**:
- 5 framework support (PyMDP, RxInfer, ActiveInference.jl, JAX, DisCoPy)
- Template-based code generation
- Comprehensive documentation generation
- POMDP-aware rendering

**Known Issues**:
- Julia code validation requires Julia installation
- Complex models may need manual code adjustments

### Roadmap
- **Next Version**: Add support for custom templates
- **Future**: Incremental rendering, code optimization passes

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [GNN Module](../gnn/AGENTS.md)
- [Execute Module](../execute/AGENTS.md)

### External Resources
- [PyMDP Documentation](https://github.com/infer-actively/pymdp)
- [RxInfer.jl Documentation](https://github.com/ReactiveBayes/RxInfer.jl)
- [ActiveInference.jl Documentation](https://github.com/ilabcode/ActiveInference.jl)

---

**Last Updated**: September 29, 2025  
**Maintainer**: GNN Pipeline Team  
**Status**: ✅ Production Ready



