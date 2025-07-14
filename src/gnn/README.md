# GNN (Generalized Notation Notation) Core Module

This module provides the core infrastructure for GNN (Generalized Notation Notation) - a standardized language for specifying Active Inference generative models.

## Overview

GNN enables researchers and practitioners to:
- Specify generative models in a standardized, machine-readable format
- Validate model specifications against formal schemas
- Parse and analyze model structures programmatically
- Export models to various simulation environments (PyMDP, RxInfer.jl)
- Generate visualizations and documentation automatically

## Module Structure

```
src/gnn/
├── __init__.py                 # Module initialization
├── README.md                   # This documentation
├── mcp.py                      # Model Context Protocol integration
├── schema_validator.py         # Validation and parsing functionality
├── gnn_schema.json            # JSON Schema definition
├── gnn_schema.yaml            # YAML Schema definition (human-readable)
├── gnn_grammar.ebnf           # Formal grammar specification
├── gnn_file_structure.md      # File structure documentation
├── gnn_punctuation.md         # Syntax punctuation guide
└── input/gnn_files/           # Example GNN files
    ├── pymdp_pomdp_agent.md
    ├── rxinfer_hidden_markov_model.md
    ├── rxinfer_multiagent_gnn.md
    └── self_driving_car_comprehensive.md
```

## Machine-Readable Schema Components

### 1. JSON Schema (`gnn_schema.json`)

Formal JSON Schema definition that provides:
- Complete structural validation rules
- Type definitions for all GNN components
- Required and optional section specifications
- Active Inference specific patterns and constraints

**Usage:**
```python
import json
import jsonschema

with open('gnn_schema.json', 'r') as f:
    schema = json.load(f)

# Validate a GNN structure (when converted to JSON)
jsonschema.validate(gnn_data, schema)
```

### 2. YAML Schema (`gnn_schema.yaml`)

Human-readable schema definition that includes:
- Detailed section descriptions and examples
- Syntax rules and validation patterns
- Active Inference naming conventions
- Validation levels and best practices
- Common error patterns and solutions

**Usage:**
```python
import yaml

with open('gnn_schema.yaml', 'r') as f:
    schema_info = yaml.safe_load(f)

required_sections = schema_info['required_sections']
syntax_rules = schema_info['syntax_rules']
```

### 3. EBNF Grammar (`gnn_grammar.ebnf`)

Extended Backus-Naur Form grammar specification that defines:
- Complete syntax rules for GNN files
- Token definitions and parsing rules
- Expression precedence and associativity
- Advanced constructs for mathematical notation

**Usage:**
Can be used with parser generators like ANTLR, PLY, or Lark to create custom parsers.

### 4. Schema Validator (`schema_validator.py`)

Python module providing:
- Complete GNN file parsing and validation
- Multiple validation levels (basic, standard, strict, research)
- Detailed error reporting and suggestions
- Active Inference compliance checking

**Usage:**
```python
from gnn.schema_validator import validate_gnn_file, parse_gnn_file

# Validate a GNN file
result = validate_gnn_file('model.md')
if result.is_valid:
    print("Valid GNN file!")
else:
    print("Errors:", result.errors)
    print("Warnings:", result.warnings)

# Parse a GNN file into structured format
parsed = parse_gnn_file('model.md')
print("Variables:", parsed.variables)
print("Connections:", parsed.connections)
```

## GNN File Structure

GNN files are Markdown-based with specific sections:

### Required Sections

1. **GNNSection**: Unique model identifier
2. **GNNVersionAndFlags**: GNN version and processing flags
3. **ModelName**: Descriptive model name
4. **ModelAnnotation**: Free-text model description
5. **StateSpaceBlock**: Variable definitions with dimensions and types
6. **Connections**: Directed/undirected edges between variables
7. **InitialParameterization**: Starting values and parameters
8. **Time**: Temporal configuration (Static/Dynamic)
9. **Footer**: Closing metadata

### Optional Sections

- **ImageFromPaper**: Visual representation of the model
- **Equations**: LaTeX mathematical relationships
- **ActInfOntologyAnnotation**: Ontology term mappings
- **ModelParameters**: Model-specific metadata
- **Signature**: Provenance and verification information

## Syntax Examples

### Variable Definitions
```
# StateSpaceBlock
A[3,3,type=float]              # 3x3 transition matrix
s_f0[2,1,type=float]           # Hidden state factor 0
o_m0[3,1,type=int]             # Observation modality 0
learning_rate[1,type=float]    # Scalar learning rate
```

### Connections
```
# Connections
A>B                            # A influences B (directed)
(A,B)-C                        # A and B associated with C (undirected)
X|Y                            # X conditional on Y
(s_f0,s_f1)>(A_m0,A_m1,A_m2)  # Multiple variables
```

### Parameters
```
# InitialParameterization
A={(1.0, 0.0), (0.0, 1.0)}     # Matrix initialization
learning_rate=0.01             # Scalar value
enabled=true                   # Boolean value
```

## Active Inference Conventions

GNN follows standard Active Inference naming patterns:

- **A matrices**: Likelihood/observation matrices `A_m0`, `A_m1`, etc.
- **B matrices**: Transition dynamics `B_f0`, `B_f1`, etc.
- **C vectors**: Preferences/goals `C_m0`, `C_m1`, etc.
- **D vectors**: Priors over initial states `D_f0`, `D_f1`, etc.
- **Hidden states**: `s_f0`, `s_f1`, etc.
- **Observations**: `o_m0`, `o_m1`, etc.
- **Actions**: `u_c0`, `u_c1`, etc.
- **Policies**: `π_c0`, `π_c1`, etc.

## Validation Levels

### Basic
- File structure validation
- Required sections present
- Basic syntax checking

### Standard (Default)
- All basic checks
- Variable definition validation
- Connection reference checking
- Parameter format validation

### Strict
- All standard checks
- Complete parameterization required
- Ontology mappings encouraged
- Mathematical consistency checking

### Research
- All strict checks
- Active Inference compliance
- Scientific reproducibility standards
- Complete provenance tracking

## Model Context Protocol (MCP) Integration

The module provides MCP tools for external integration:

### Available Tools

1. **get_gnn_documentation**: Retrieve documentation files
2. **validate_gnn_content**: Validate GNN content against schema
3. **get_gnn_schema_info**: Get comprehensive schema information

### Usage with MCP

```python
from gnn.mcp import register_tools

# Register GNN tools with MCP server
register_tools(mcp_server)

# Tools are now available via MCP protocol
```

## Examples

The `input/gnn_files/` directory contains complete GNN files demonstrating:

- **PyMDP POMDP Agent**: Multi-factor agent with observation modalities
- **RxInfer Hidden Markov Model**: Basic HMM for RxInfer.jl
- **Multi-agent Trajectory Planning**: Complex multi-agent system
- **Comprehensive Self-Driving Car**: Large-scale autonomous vehicle model

## Best Practices

1. **Use descriptive variable names** that reflect their meaning
2. **Include comments** (`###`) to explain complex variables
3. **Follow Active Inference conventions** for matrix naming
4. **Provide ontology mappings** for interoperability
5. **Include equations** to clarify mathematical relationships
6. **Validate files** before sharing or publication
7. **Use consistent formatting** and indentation

## Error Handling

Common validation errors and solutions:

### Missing Required Section
```
Error: Required section missing: StateSpaceBlock
Solution: Add ## StateSpaceBlock header with variable definitions
```

### Invalid Variable Name
```
Error: Invalid variable name format: 2A
Solution: Variable names must start with letter/underscore: A_2
```

### Undefined Variable in Connection
```
Error: Connection references undefined variable: B
Solution: Define B in StateSpaceBlock before referencing in Connections
```

## Integration with Pipeline

This module integrates with the broader GNN pipeline:

1. **Step 1 (1_gnn.py)**: Uses this module for file discovery and basic parsing
2. **Step 4 (4_type_checker.py)**: Uses validation for type checking
3. **Step 5 (5_export.py)**: Uses parsed structures for format conversion
4. **Step 6 (6_visualization.py)**: Uses connections for graph visualization
5. **Step 9 (9_render.py)**: Uses parsed models for code generation

## Development and Extension

### Adding New Validation Rules

1. Extend `schema_validator.py` with new validation methods
2. Update JSON/YAML schemas with new constraints
3. Add test cases for new validation scenarios
4. Update documentation with new requirements

### Supporting New Backends

1. Add backend-specific validation in `_validate_active_inference_compliance`
2. Update ModelParameters section specification
3. Add examples demonstrating backend-specific features
4. Update export pipeline for new backend support

## Performance Considerations

- **Schema loading**: Schemas are cached after first load
- **Validation**: Use appropriate validation level for your use case
- **Large files**: Consider streaming validation for very large models
- **Memory usage**: Parsed structures are kept in memory during validation

## License and Citation

This implementation follows the GNN specification v1.0 and is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 