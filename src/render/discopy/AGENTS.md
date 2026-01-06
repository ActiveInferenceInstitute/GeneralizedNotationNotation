# DisCoPy Render Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Code generation for DisCoPy (categorical diagrams) framework from GNN specifications

**Parent Module**: Render Module (Step 11: Code rendering)

**Category**: Framework Code Generation / DisCoPy

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to DisCoPy categorical diagrams
2. Generate string diagram representations of compositional models
3. Create DisCoPy code for categorical quantum computing and NLP
4. Handle DisCoPy-specific optimizations and diagram composition
5. Support functorial semantics and monoidal categories

### Key Capabilities
- String diagram generation from GNN models
- Categorical composition and functor application
- Quantum circuit representation
- Natural language processing diagrams
- DisCoPy-specific template management
- Diagram optimization and simplification

---

## API Reference

### Public Functions

#### `generate_discopy_code(model_data: Dict[str, Any], output_path: Optional[str] = None) -> str`
**Description**: Generate DisCoPy code from GNN model data

**Parameters**:
- `model_data` (Dict): GNN model data dictionary
- `output_path` (Optional[str]): Output file path (optional)

**Returns**: Generated DisCoPy code as string

**Example**:
```python
from render.discopy import generate_discopy_code

# Generate DisCoPy code
discopy_code = generate_discopy_code(model_data)

# Save to file
with open("categorical_diagram.py", "w") as f:
    f.write(discopy_code)
```

#### `convert_gnn_to_discopy(model_data: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Convert GNN model data to DisCoPy-compatible format

**Parameters**:
- `model_data` (Dict): GNN model data

**Returns**: DisCoPy-compatible model structure

#### `create_discopy_diagram(model_structure: Dict[str, Any], config: Dict[str, Any]) -> str`
**Description**: Create DisCoPy diagram implementation

**Parameters**:
- `model_structure` (Dict): Model structure data
- `config` (Dict): DisCoPy configuration options

**Returns**: DisCoPy diagram code

#### `generate_discopy_visualization(model_data: Dict[str, Any], config: Dict[str, Any]) -> str`
**Description**: Generate DisCoPy diagram with visualization

**Parameters**:
- `model_data` (Dict): GNN model data
- `config` (Dict): Visualization configuration

**Returns**: DisCoPy code with visualization

---

## Dependencies

### Required Dependencies
- `discopy` - DisCoPy categorical computing library
- `matplotlib` - Diagram visualization
- `numpy` - Numerical computations

### Optional Dependencies
- `pytket` - Quantum circuit compilation (fallback: basic diagrams)
- `lambeq` - NLP diagram processing (fallback: basic composition)

### Internal Dependencies
- `render.renderer` - Base rendering functionality
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### DisCoPy Configuration
```python
DISCOPY_CONFIG = {
    'category': 'quantum',            # Category type (quantum, classical, nlp)
    'backend': 'pytket',              # Backend for execution
    'optimization': True,             # Diagram optimization
    'visualization': True,            # Generate visualizations
    'simplification': 'eager',        # Diagram simplification strategy
    'composition': 'sequential',      # Composition strategy
    'functor': 'identity',            # Functor application
    'evaluation': 'tensor'            # Evaluation method
}
```

### Model Conversion Configuration
```python
CONVERSION_CONFIG = {
    'node_mapping': 'boxes',          # Map GNN nodes to diagram boxes
    'edge_mapping': 'wires',          # Map connections to wires
    'type_mapping': 'categorical',    # Type system mapping
    'composition_order': 'left-to-right',  # Diagram composition order
    'normalization': True,            # Diagram normalization
    'typing': 'strict'                # Type checking strictness
}
```

### Visualization Configuration
```python
VISUALIZATION_CONFIG = {
    'diagram_style': 'planar',        # Diagram drawing style
    'color_scheme': 'categorical',    # Color scheme for visualization
    'layout_engine': 'graphviz',      # Layout algorithm
    'output_format': 'png',           # Visualization output format
    'show_types': True,               # Display type annotations
    'show_labels': True,              # Display box labels
    'scale_factor': 1.0               # Diagram scaling
}
```

---

## Usage Examples

### Basic DisCoPy Code Generation
```python
from render.discopy import generate_discopy_code

# Example GNN model data for categorical representation
model_data = {
    "variables": {
        "subject": {"type": "noun", "category": "N"},
        "verb": {"type": "verb", "category": "S\\N"},
        "object": {"type": "noun", "category": "N"}
    },
    "connections": [
        {"from": "subject", "to": "verb", "type": "application"},
        {"from": "verb", "to": "object", "type": "application"}
    ],
    "constraints": [
        {"type": "type_check", "requirement": "well_typed"}
    ]
}

# Generate DisCoPy code
discopy_code = generate_discopy_code(model_data)
print(discopy_code[:500])  # Print first 500 characters
```

### Diagram Creation and Visualization
```python
from render.discopy import create_discopy_diagram, generate_discopy_visualization

# Create DisCoPy diagram
diagram_config = {
    'category': 'quantum',
    'visualization': True,
    'optimization': True
}

diagram_code = create_discopy_diagram(model_data, diagram_config)

# Generate with visualization
viz_code = generate_discopy_visualization(model_data, diagram_config)

# Save to files
with open("quantum_diagram.py", "w") as f:
    f.write(diagram_code)

with open("diagram_visualization.py", "w") as f:
    f.write(viz_code)
```

### Model Conversion
```python
from render.discopy import convert_gnn_to_discopy

# Convert GNN model to DisCoPy format
discopy_model = convert_gnn_to_discopy(model_data)

print("DisCoPy Model Structure:")
print(f"Boxes: {len(discopy_model['boxes'])}")
print(f"Wires: {len(discopy_model['wires'])}")
print(f"Category: {discopy_model['category']}")
print(f"Types: {list(discopy_model['types'].keys())}")
```

### Categorical Composition
```python
from render.discopy import generate_discopy_composition

# Generate composition example
composition_config = {
    'diagrams': ['diagram1', 'diagram2', 'diagram3'],
    'composition_type': 'monoidal',
    'output_type': 'quantum_circuit'
}

composition_code = generate_discopy_composition(model_data, composition_config)

# This generates code for composing multiple diagrams
print(composition_code)
```

---

## Categorical Concepts Mapping

### GNN to DisCoPy Mapping
- **Variables → Boxes**: GNN variables become diagram boxes with types
- **Connections → Wires**: Relationships become typed wires between boxes
- **Constraints → Functors**: Constraints become functor applications
- **Composition → Diagrams**: Model structure becomes diagram composition
- **Types → Categories**: GNN types become categorical types

### DisCoPy Components Generated
- **Diagram**: Main string diagram structure
- **Box**: Typed computational units
- **Wire**: Typed connections between boxes
- **Category**: Mathematical category structure
- **Functor**: Structure-preserving mappings

---

## Output Specification

### Output Products
- `*_discopy_diagram.py` - DisCoPy diagram code
- `*_discopy_visualization.py` - Visualization code
- `*_discopy_evaluation.py` - Evaluation and execution code
- `diagram_visualization.png` - Diagram visualization image

### Output Directory Structure
```
output/11_render_output/
├── model_name_discopy_diagram.py
├── model_name_discopy_visualization.py
├── model_name_discopy_evaluation.py
└── diagram_visualization.png
```

### Generated Script Structure
```python
# Generated DisCoPy script structure
import discopy as dp
from discopy import quantum
from discopy.drawing import draw

# Define types and boxes
n, s = map(dp.types.Ob, "ns")
subject = dp.Box("subject", dp.types.Ty(), n)
verb = dp.Box("verb", n, s @ n.l)
obj = dp.Box("object", dp.types.Ty(), n)

# Compose diagram
sentence = subject @ verb @ obj
diagram = sentence >> dp.Id(n) @ dp.Cup(n, n.l) @ dp.Id(n)

# Draw and save visualization
diagram.draw(path="diagram.png")

# Evaluate diagram
result = dp.tensor.eval(diagram)

# Quantum compilation (if applicable)
if hasattr(diagram, 'to_pytket'):
    circuit = diagram.to_pytket()
    # Execute on quantum backend
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 1-2 seconds per model
- **Memory**: 100-250MB depending on diagram complexity
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Code Generation**: < 500ms
- **Template Processing**: < 1s
- **Visualization**: < 1s
- **Validation**: < 500ms

### Optimization Notes
- DisCoPy generation includes diagram optimization
- Memory usage depends on diagram size and type complexity
- Generated code is optimized for categorical composition

---

## Error Handling

### DisCoPy Generation Errors
1. **Type Mismatch**: Incompatible categorical types
2. **Composition Error**: Invalid diagram composition
3. **Backend Error**: Unsupported backend operations

### Recovery Strategies
- **Type Validation**: Comprehensive type checking before generation
- **Composition Verification**: Diagram composition validation
- **Backend Fallback**: Fallback to basic evaluation when advanced backends fail

### Error Examples
```python
try:
    discopy_code = generate_discopy_code(model_data)
except DisCoPyGenerationError as e:
    logger.error(f"DisCoPy generation failed: {e}")
    # Fallback to basic diagram
    discopy_code = generate_basic_discopy_template(model_data)
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
- `execute.discopy` - DisCoPy execution module
- `tests.test_render_discopy*` - DisCoPy-specific tests

### Data Flow
```
GNN Model → DisCoPy Conversion → Type Assignment → Diagram Composition → Visualization → Validation → Output
```

---

## Testing

### Test Files
- `src/tests/test_render_discopy_integration.py` - Integration tests
- `src/tests/test_render_discopy_generation.py` - Code generation tests
- `src/tests/test_render_discopy_validation.py` - Validation tests

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

### Key Test Scenarios
1. DisCoPy code generation from various GNN models
2. Generated Python code syntax validation
3. Categorical type system correctness
4. Diagram composition and evaluation
5. Error handling for invalid categorical structures

### Test Commands
```bash
# Run DisCoPy-specific tests
pytest src/tests/test_render_discopy*.py -v

# Run with coverage
pytest src/tests/test_render_discopy*.py --cov=src/render/discopy --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `render.generate_discopy` - Generate DisCoPy diagram code
- `render.convert_to_discopy` - Convert GNN to DisCoPy format
- `render.visualize_discopy` - Generate DisCoPy visualizations
- `render.validate_discopy` - Validate DisCoPy diagram structure

### Tool Endpoints
```python
@mcp_tool("render.generate_discopy")
def generate_discopy_tool(model_data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate DisCoPy diagram code from GNN model"""
    return generate_discopy_code(model_data, **config)
```

---

## Categorical Computing Features

### Diagram Types Supported
- **String Diagrams**: Compositional structure representation
- **Quantum Circuits**: Quantum computing representations
- **NLP Diagrams**: Natural language processing diagrams
- **Control Flow**: Conditional and iterative structures

### Optimization Strategies
- **Diagram Simplification**: Automatic diagram reduction
- **Type Inference**: Automatic type assignment and checking
- **Composition Optimization**: Efficient diagram composition
- **Evaluation Optimization**: Optimized tensor evaluation

---

## Development Guidelines

### Adding New DisCoPy Features
1. Update conversion logic in `discopy_renderer.py`
2. Add new diagram types and compositions
3. Update visualization options
4. Add comprehensive tests

### Template Management
- Templates are implemented as functions in renderer files
- Use dynamic code generation for diagram construction
- Maintain type safety and categorical correctness
- Include visualization and evaluation code

---

## Troubleshooting

### Common Issues

#### Issue 1: "Categorical type mismatch"
**Symptom**: DisCoPy diagram creation fails with type errors
**Cause**: Incompatible GNN types or connection patterns
**Solution**: Validate categorical type system before conversion

#### Issue 2: "Diagram composition failed"
**Symptom**: Cannot compose diagrams due to structure incompatibility
**Cause**: Invalid composition order or missing connections
**Solution**: Check diagram composition rules and reorder if necessary

#### Issue 3: "Backend evaluation failed"
**Symptom**: Diagram evaluation fails on specific backend
**Cause**: Backend-specific limitations or unsupported operations
**Solution**: Switch to compatible backend or simplify diagram

### Debug Mode
```python
# Enable debug output for DisCoPy generation
result = generate_discopy_code(model_data, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete DisCoPy code generation pipeline
- Categorical diagram composition and visualization
- Multiple backend support (quantum, classical, NLP)
- Type system validation and optimization
- Comprehensive error handling and recovery
- MCP tool integration

**Known Limitations**:
- Complex quantum circuits may require manual optimization
- Very large diagrams may impact visualization performance
- Some advanced categorical features require manual implementation

### Roadmap
- **Next Version**: Enhanced quantum circuit support
- **Future**: Automatic diagram optimization
- **Advanced**: Integration with categorical databases and languages

---

## References

### Related Documentation
- [Render Module](../../render/AGENTS.md) - Parent render module
- [DisCoPy Documentation](https://discopy.readthedocs.io/) - Official DisCoPy docs
- [Categorical Quantum Mechanics](https://arxiv.org/abs/quant-ph/0510032) - Theoretical foundation

### External Resources
- [String Diagrams](https://en.wikipedia.org/wiki/String_diagram)
- [Category Theory](https://en.wikipedia.org/wiki/Category_theory)
- [Categorical Compositional Models](https://arxiv.org/abs/1905.06547)

---

**Last Updated**: 2025-12-30
**Maintainer**: Render Module Team
**Status**: ✅ Production Ready




