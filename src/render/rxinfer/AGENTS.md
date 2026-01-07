# RxInfer.jl Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for RxInfer.jl (Julia) probabilistic programming framework simulations from GNN specifications

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / RxInfer.jl

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to RxInfer.jl simulation code
2. Generate reactive probabilistic models in Julia
3. Create TOML configuration files for RxInfer.jl
4. Handle RxInfer.jl-specific optimizations and reactive constraints
5. Support RxInfer.jl's message-passing inference algorithms

### Key Capabilities
- Reactive probabilistic model generation
- Message-passing inference setup
- TOML configuration file generation
- RxInfer.jl-specific template management
- Reactive constraint implementation
- Julia code generation with proper syntax

---

## API Reference

### Public Functions

#### `generate_rxinfer_code(model_data: Dict[str, Any], output_path: Optional[Union[str, Path]] = None, **kwargs) -> str`
**Description**: Generate RxInfer.jl simulation code from GNN model data.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data dictionary with variables, connections, matrices
- `output_path` (Optional[Union[str, Path]]): Output file path (optional, if provided code is also written to file)
- `include_toml` (bool, optional): Generate TOML configuration file (default: True)
- `template_type` (str, optional): Template type ("minimal", "simple", "full") (default: "simple")
- `**kwargs`: Additional RxInfer.jl generation options

**Returns**: `str` - Generated RxInfer.jl code as string

**Example**:
```python
from render.rxinfer import generate_rxinfer_code
from pathlib import Path

# Generate RxInfer.jl code
rxinfer_code = generate_rxinfer_code(
    model_data=parsed_gnn_model,
    output_path=Path("output/simulation.jl"),
    include_toml=True,
    template_type="simple"
)

# Code is also saved to file if output_path provided
```

#### `generate_rxinfer_toml_config(model_data: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Generate TOML configuration for RxInfer.jl simulation.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data
- `config` (Dict[str, Any], optional): Simulation configuration (default: {})
- `project_name` (str, optional): Julia project name (default: "GNN_Model")
- `**kwargs`: Additional TOML generation options

**Returns**: `str` - TOML configuration as string

#### `convert_gnn_to_rxinfer(model_data: Dict[str, Any], **kwargs) -> Dict[str, Any]`
**Description**: Convert GNN model data to RxInfer.jl-compatible format.

**Parameters**:
- `model_data` (Dict[str, Any]): GNN model data with variables, connections, matrices
- `validate` (bool, optional): Validate RxInfer.jl compatibility (default: True)
- `**kwargs`: Additional conversion options

**Returns**: `Dict[str, Any]` - RxInfer.jl-compatible model structure with:
- `variables` (List[Dict]): Variable definitions
- `factors` (List[Dict]): Factor graph definitions
- `constraints` (List[Dict]): Reactive constraints
- `inference_config` (Dict): Inference algorithm configuration

#### `create_rxinfer_model(model_structure: Dict[str, Any], config: Dict[str, Any] = None, **kwargs) -> str`
**Description**: Create RxInfer.jl model implementation code.

**Parameters**:
- `model_structure` (Dict[str, Any]): RxInfer.jl-compatible model structure
- `config` (Dict[str, Any], optional): RxInfer.jl configuration options (default: {})
- `inference_algorithm` (str, optional): Inference algorithm ("VMP", "BP", "EP") (default: "VMP")
- `**kwargs`: Additional model generation options

**Returns**: `str` - RxInfer.jl model code as string

---

## Dependencies

### Required Dependencies
- `RxInfer.jl` - RxInfer.jl Julia package
- `ReactiveMP.jl` - Reactive message passing library
- `GraphPPL.jl` - Probabilistic programming DSL
- `TOML.jl` - TOML file parsing (for Julia)

### Optional Dependencies
- `Plots.jl` - Visualization support (fallback: no plotting)
- `DataFrames.jl` - Data manipulation (fallback: basic arrays)

### Internal Dependencies
- `render.renderer` - Base rendering functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### RxInfer.jl Configuration
```python
RXINFER_CONFIG = {
    'inference_engine': 'reactive',    # Reactive inference engine
    'optimization': 'auto',           # Automatic optimization
    'constraints': 'default',         # Default reactive constraints
    'scheduler': 'Asynchronous',      # Inference scheduler
    'iterations': 100,                # Maximum iterations
    'tolerance': 1e-6,                # Convergence tolerance
    'autoupdates': True,              # Automatic model updates
    'meta': True                      # Use meta-programming
}
```

### Model Conversion Configuration
```python
CONVERSION_CONFIG = {
    'node_mapping': 'reactive',       # Map GNN nodes to reactive nodes
    'edge_mapping': 'message',        # Map connections to message passing
    'factor_mapping': 'constraint',   # Map constraints to factors
    'prior_mapping': 'initial',       # Map priors to initial messages
    'likelihood_mapping': 'observation'  # Map likelihoods to observations
}
```

### TOML Configuration
```python
TOML_CONFIG = {
    'simulation': {
        'trials': 100,                # Number of trials
        'length': 50,                 # Trial length
        'seed': 42                    # Random seed
    },
    'inference': {
        'engine': 'ReactiveMP',
        'iterations': 100,
        'tolerance': 1e-6
    },
    'output': {
        'save_results': True,
        'plot_results': False,
        'verbose': False
    }
}
```

---

## Usage Examples

### Basic RxInfer.jl Code Generation
```python
from render.rxinfer import generate_rxinfer_code

# Example GNN model data
model_data = {
    "variables": {
        "x": {"domain": ["low", "high"], "type": "categorical"},
        "y": {"domain": ["small", "large"], "type": "categorical"},
        "z": {"domain": ["near", "far"], "type": "categorical"}
    },
    "connections": [
        {"from": "x", "to": "y", "type": "transition"},
        {"from": "y", "to": "z", "type": "emission"}
    ]
}

# Generate RxInfer.jl code
rxinfer_code = generate_rxinfer_code(model_data)
print(rxinfer_code[:500])  # Print first 500 characters
```

### TOML Configuration Generation
```python
from render.rxinfer import generate_rxinfer_toml_config

# Configuration for RxInfer.jl
config = {
    'inference': {
        'iterations': 200,
        'tolerance': 1e-8
    },
    'simulation': {
        'trials': 50,
        'length': 100
    }
}

# Generate TOML config
toml_config = generate_rxinfer_toml_config(model_data, config)

# Save to file
with open("config.toml", "w") as f:
    f.write(toml_config)
```

### Model Conversion
```python
from render.rxinfer import convert_gnn_to_rxinfer

# Convert GNN model to RxInfer.jl format
rxinfer_model = convert_gnn_to_rxinfer(model_data)

print("RxInfer.jl Model Structure:")
print(f"Variables: {list(rxinfer_model['variables'].keys())}")
print(f"Factors: {list(rxinfer_model['factors'].keys())}")
print(f"Constraints: {len(rxinfer_model['constraints'])}")
```

### Complete Simulation Generation
```python
from render.rxinfer import create_rxinfer_model

# Create complete RxInfer.jl model
model_config = {
    'constraints': 'default',
    'inference': 'reactive',
    'meta': True
}

model_code = create_rxinfer_model(model_data, model_config)

# Save model code
with open("model.jl", "w") as f:
    f.write(model_code)
```

---

## RxInfer.jl Concepts Mapping

### GNN to RxInfer.jl Mapping
- **Variables → Random Variables**: GNN variables become RxInfer.jl random variables
- **Connections → Factors**: Relationships become probabilistic factors
- **Constraints → Constraints**: GNN constraints become RxInfer.jl constraints
- **Observations → Data**: Observed variables become data streams
- **Inference → Message Passing**: Inference becomes reactive message passing

### RxInfer.jl Components Generated
- **@model macro**: Probabilistic model definition
- **@constraints**: Model constraints and approximations
- **@meta**: Meta-programming specifications
- **inference()**: Reactive inference specification
- **subscribe!()**: Result subscription and callbacks

---

## Output Specification

### Output Products
- `*_rxinfer_simulation.jl` - Complete RxInfer.jl simulation scripts
- `*_rxinfer_model.jl` - RxInfer.jl model definition files
- `config.toml` - TOML configuration file
- `*_rxinfer_inference.jl` - Inference specification files

### Output Directory Structure
```
output/11_render_output/
├── model_name_rxinfer_simulation.jl
├── model_name_rxinfer_model.jl
├── config.toml
└── model_name_rxinfer_inference.jl
```

### Generated Script Structure
```julia
# Generated RxInfer.jl simulation script structure
using RxInfer
using GraphPPL
using ReactiveMP

# Model definition
@model function my_model()
    # Variable declarations
    x = datavar(Float64)
    y = randomvar(Float64)

    # Factor definitions
    x ~ Normal(μₓ, σₓ)
    y ~ Normal(x, σ_y)

    # Observations
    y_observed ~ Normal(y, σ_observation)
end

# Constraints and meta
@constraints function my_constraints()
    # Constraint specifications
end

@meta function my_meta()
    # Meta-programming specifications
end

# Inference specification
model = my_model()
result = inference(
    model = model,
    data = (y_observed = observations,),
    constraints = my_constraints(),
    meta = my_meta(),
    iterations = 100
)

# Results processing
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 2-4 seconds per model
- **Memory**: 100-300MB depending on model complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Code Generation**: < 1s
- **TOML Generation**: < 500ms
- **Template Processing**: < 2s
- **Validation**: < 500ms

### Optimization Notes
- RxInfer.jl generation includes reactive optimizations
- TOML configuration allows runtime parameter tuning
- Generated Julia code is optimized for message-passing efficiency

---

## Error Handling

### RxInfer.jl Generation Errors
1. **Invalid Model Structure**: GNN model cannot be mapped to RxInfer.jl
2. **Julia Syntax Errors**: Generated code has syntax issues
3. **TOML Configuration Errors**: Invalid configuration parameters

### Recovery Strategies
- **Model Validation**: Comprehensive pre-generation validation
- **Template Fallback**: Use simpler templates for complex models
- **Syntax Checking**: Julia syntax validation of generated code

### Error Examples
```python
try:
    rxinfer_code = generate_rxinfer_code(model_data)
except RxInferGenerationError as e:
    logger.error(f"RxInfer.jl generation failed: {e}")
    # Fallback to minimal template
    rxinfer_code = generate_minimal_rxinfer_template(model_data)
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
- `execute.rxinfer` - RxInfer.jl execution module
- `tests.test_render_rxinfer*` - RxInfer.jl-specific tests

### Data Flow
```
GNN Model → RxInfer.jl Conversion → Template Application → Julia Code Generation → TOML Config → Validation → Output
```

---

## Testing

### Test Files
- `src/tests/test_render_rxinfer_integration.py` - Integration tests
- `src/tests/test_render_rxinfer_generation.py` - Code generation tests
- `src/tests/test_render_rxinfer_validation.py` - Validation tests

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

### Key Test Scenarios
1. RxInfer.jl code generation from various GNN models
2. Generated Julia code syntax validation
3. TOML configuration generation and parsing
4. Reactive constraint implementation
5. Error handling for invalid models

### Test Commands
```bash
# Run RxInfer.jl-specific tests
pytest src/tests/test_render_rxinfer*.py -v

# Run with coverage
pytest src/tests/test_render_rxinfer*.py --cov=src/render/rxinfer --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `render.generate_rxinfer` - Generate RxInfer.jl simulation code
- `render.convert_to_rxinfer` - Convert GNN to RxInfer.jl format
- `render.generate_toml` - Generate TOML configuration
- `render.validate_rxinfer` - Validate RxInfer.jl model structure

### Tool Endpoints
```python
@mcp_tool("render.generate_rxinfer")
def generate_rxinfer_tool(model_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate RxInfer.jl simulation code from GNN model"""
    return generate_rxinfer_code(model_data, **config)
```

---

## RxInfer.jl-Specific Features

### Reactive Inference Implementation
- **Message Passing**: Belief propagation through factor graphs
- **Reactive Updates**: Automatic updates when new data arrives
- **Constraint Programming**: Declarative constraint specification
- **Meta-Programming**: Compile-time optimizations

### Optimization Strategies
- **Graph Compilation**: Efficient factor graph compilation
- **Message Scheduling**: Optimized message-passing schedules
- **Memory Management**: Efficient memory usage for large models
- **Parallel Inference**: Parallel message passing when possible

---

## Development Guidelines

### Adding New RxInfer.jl Features
1. Update model conversion logic in `rxinfer_renderer.py`
2. Add new templates in template files (minimal_template.jl, simple_template.jl)
3. Update TOML configuration generation
4. Add comprehensive tests

### Template Management
- Templates are stored as .jl files in the rxinfer directory
- Use string templating for dynamic Julia code generation
- Maintain template modularity for different RxInfer.jl features
- Include proper Julia syntax and imports

---

## Troubleshooting

### Common Issues

#### Issue 1: "Julia syntax error in generated code"
**Symptom**: RxInfer.jl simulation fails with syntax errors
**Cause**: Invalid Julia code generation or template issues
**Solution**: Validate generated code syntax before execution

#### Issue 2: "TOML configuration parsing failed"
**Symptom**: Configuration file cannot be parsed
**Cause**: Invalid TOML structure or parameters
**Solution**: Validate TOML generation and use default values

#### Issue 3: "Reactive inference not converging"
**Symptom**: Inference fails to converge within iterations
**Cause**: Poor model specification or inappropriate parameters
**Solution**: Adjust inference parameters or simplify constraints

### Debug Mode
```python
# Enable debug output for RxInfer.jl generation
result = generate_rxinfer_code(model_data, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete RxInfer.jl code generation pipeline
- Reactive probabilistic model implementation
- TOML configuration file generation
- Template-based Julia code generation
- Comprehensive error handling and validation
- MCP tool integration

**Known Limitations**:
- Complex hierarchical models may require manual optimization
- Very large models may impact compilation time
- Some advanced RxInfer.jl features require manual implementation

### Roadmap
- **Next Version**: Enhanced reactive constraint support
- **Future**: Automatic model optimization
- **Advanced**: Integration with RxInfer.jl's latest reactive features

---

## References

### Related Documentation
- [Render Module](../../render/AGENTS.md) - Parent render module
- [RxInfer.jl Documentation](https://rxinfer.ml/) - Official RxInfer.jl docs
- [ReactiveMP.jl](https://github.com/biaslab/ReactiveMP.jl) - Reactive message passing

### External Resources
- [Probabilistic Programming](https://en.wikipedia.org/wiki/Probabilistic_programming)
- [Message Passing](https://en.wikipedia.org/wiki/Belief_propagation)
- [Factor Graphs](https://en.wikipedia.org/wiki/Factor_graph)

---

**Last Updated**: 2026-01-07
**Maintainer**: Render Module Team
**Status**: ✅ Production Ready




