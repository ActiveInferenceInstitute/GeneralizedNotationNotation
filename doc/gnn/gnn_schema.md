# GNN Schema Specification

Complete specification for GNN syntax parsing and validation.

## Core Schema Components

### Variable Declaration
```gnn
<name>[<dimensions>,type=<type>]
```

**Implementation:** `src/gnn/parser.py:parse_variable_declaration()`

**Schema Rules:**
- `<name>`: `[a-zA-Z_][a-zA-Z0-9_]*` (identifier pattern)
- `<dimensions>`: comma-separated integers `\d+(,\d+)*`
- `<type>`: `int|float|double|bool|string`

**Examples from actinf_pomdp_agent.md:**
```gnn
A[3,3,type=float]           # 3×3 matrix, float type
B[3,3,3,type=float]         # 3×3×3 tensor, float type  
s[3,1,type=float]           # 3D vector, float type
o[3,1,type=int]             # 3D vector, integer type
t[1,type=int]               # scalar, integer type
```

### Connection Syntax
```gnn
<source>-<target>    # Undirected connection
<source>><target>    # Directed connection
```

**Implementation:** `src/gnn/parser.py:parse_connection()`

**Schema Rules:**
- `<source>`,`<target>`: variable names or compound expressions
- Operators: `-` (undirected), `>` (directed)

**Examples from actinf_pomdp_agent.md:**
```gnn
D>s          # D causes s (directed)
s-A          # s relates to A (undirected)
A-o          # A relates to o (undirected)
π>u          # π causes u (directed)
```

### Section Headers
```gnn
## <SectionName>
```

**Implementation:** `src/gnn/parser.py:parse_section_header()`

**Required Sections:**
- `## StateSpaceBlock` - Variable declarations
- `## Connections` - Connection specifications
- `## InitialParameterization` - Parameter values

**Optional Sections:**
- `## ModelName` - Model identifier
- `## ModelAnnotation` - Description
- `## Equations` - Mathematical relations
- `## ActInfOntologyAnnotation` - Semantic mappings

## Round-Trip Data Flow

### 1. Parse: GNN → JSON
**Entry Point:** `src/3_gnn.py:process_gnn_multi_format()`
**Core Method:** `src/gnn/multi_format_processor.py`

Input: `actinf_pomdp_agent.md`
```gnn
A[3,3,type=float]
B[3,3,3,type=float]
D>s
```

Output: `output/3_gnn_output/parsed_actinf_pomdp_agent.json`
```json
{
  "variables": [
    {"name": "A", "dimensions": [3,3], "type": "float"},
    {"name": "B", "dimensions": [3,3,3], "type": "float"}
  ],
  "connections": [
    {"source": ["D"], "target": ["s"], "type": "directed"}
  ]
}
```

### 2. Validate: JSON → Typed JSON
**Entry Point:** `src/5_type_checker.py:analyze_variable_types()`
**Core Method:** `src/type_checker/analysis_utils.py`

Applies type constraints and dimensional analysis:
```json
{
  "variables": [...],
  "type_analysis": {
    "variable_count": 7,
    "type_distribution": {"float": 5, "int": 2},
    "dimensional_complexity": "3D_TENSOR"
  }
}
```

### 3. Export: JSON → Multiple Formats
**Entry Point:** `src/7_export.py:process_export()`
**Core Methods:** `src/export/`

Produces:
- GraphML: `output/7_export_output/actinf_pomdp_agent.graphml`
- GEXF: `output/7_export_output/actinf_pomdp_agent.gexf` 
- XML: `output/7_export_output/actinf_pomdp_agent.xml`
- Pickle: `output/7_export_output/actinf_pomdp_agent.pkl`

### 4. Render: JSON → Framework Code
**Entry Point:** `src/11_render.py:process_render()`
**Core Methods:** `src/render/`

Framework targets:
- **PyMDP**: `src/render/pymdp/` → `.py` files
- **RxInfer.jl**: `src/render/rxinfer/` → `.jl` files
- **ActiveInference.jl**: `src/render/activeinference/` → `.jl` files
- **DisCoPy**: `src/render/discopy/` → categorical diagrams

## Core Method Locations (Actual Implementation)

### Parsing Pipeline (Step 3: GNN Processing)
```
src/3_gnn.py (thin orchestrator)
├── src/gnn/multi_format_processor.py (main processor)
├── src/gnn/schema_validator.py
│   └── GNNParser (line 54-89)
│       ├── SECTION_PATTERN (line 58)
│       ├── VARIABLE_PATTERN (line 59) 
│       ├── CONNECTION_PATTERN (line 60)
│       └── PARAMETER_PATTERN (line 62)
├── src/gnn/parser.py
│   └── GNNParsingSystem (line 72-173)
│       ├── _detect_format() (line 107)
│       └── _basic_parser() (line 120)
└── src/gnn/parsers/
    ├── markdown_parser.py (MarkdownGNNParser)
    ├── python_parser.py (PythonGNNParser, line 25-352)
    ├── lean_parser.py (LeanGNNParser)
    ├── protobuf_parser.py (ProtobufGNNParser)
    └── unified_parser.py (UnifiedGNNParser)
```

### Type Analysis (Step 5: Type Checking)
```
src/5_type_checker.py (thin orchestrator)
└── src/type_checker/
    ├── analysis_utils.py (line 1-62)
    │   ├── analyze_variable_types() (line 13)
    │   ├── analyze_connections() 
    │   └── estimate_computational_complexity()
    ├── checker.py 
    │   └── GNNTypeChecker (line 174-268)
    │       ├── check_file() (line 232)
    │       ├── _check_required_sections()
    │       └── _collect_variable_analysis()
    └── processor.py
        └── GNNTypeChecker (line 20-261)
            ├── _validate_type() (line 191)
            └── _analyze_types() (line 232)
```

### Visualization Pipeline (Steps 8 & 9)
```
src/8_visualization.py (thin orchestrator)
└── src/visualization/
    ├── visualizer.py (line 1-73)
    │   └── GNNVisualizer (line 66)
    ├── matrix_visualizer.py 
    │   ├── MatrixVisualizer (line 40)
    │   └── generate_matrix_visualizations() (line 941)
    ├── processor.py 
    │   ├── parse_matrix_data() (line 367)
    │   ├── generate_matrix_visualizations() (line 403)
    │   └── generate_network_visualizations() (line 523)
    └── __init__.py (safe imports with fallbacks, line 15-47)

src/9_advanced_viz.py (thin orchestrator)
└── src/advanced_visualization/
    └── visualizer.py
        └── AdvancedVisualizer (line 34)
```

### Export Pipeline (Step 7: Multi-format Export)  
```
src/7_export.py (thin orchestrator)
└── src/export/
    └── [Export modules - locations to be documented]
```

### Render Pipeline (Step 11: Code Generation)
```
src/11_render.py (thin orchestrator)  
└── src/render/
    └── [Render modules - locations to be documented]
```

## Cross-References

### Data Dependencies
- Step 3 (GNN) → Step 5 (Type Checker): `parsed_*.json`
- Step 5 (Type Checker) → Step 7 (Export): `type_check_results.json`  
- Step 3 (GNN) → Step 8 (Visualization): `parsed_*.json`
- Step 3 (GNN) → Step 11 (Render): `parsed_*.json`
- Step 11 (Render) → Step 12 (Execute): generated framework code

### Schema Validation Chain
1. **Lexical**: `src/gnn/lexer.py` - tokenization
2. **Syntactic**: `src/gnn/parser.py` - AST construction  
3. **Semantic**: `src/type_checker/analysis_utils.py` - type validation
4. **Ontological**: `src/ontology/processor.py` - domain validation

### Framework Integration Points
- **PyMDP**: Matrices → `pymdp.Agent(A=A, B=B, C=C, D=D)`
- **RxInfer.jl**: Probabilistic → `@model function gnn_model()`
- **DisCoPy**: Categories → `Diagram` objects with morphisms
- **JAX**: Arrays → `jax.numpy` optimized computations

## Validation Schema

### Variable Validation
```python
# src/type_checker/analysis_utils.py:validate_variable()
def validate_variable(var):
    assert var['name'].isidentifier()
    assert all(d > 0 for d in var['dimensions'])
    assert var['type'] in ['int', 'float', 'double', 'bool']
```

### Connection Validation  
```python
# src/type_checker/analysis_utils.py:validate_connection()
def validate_connection(conn, variables):
    assert conn['source'] in [v['name'] for v in variables]
    assert conn['target'] in [v['name'] for v in variables]
    assert conn['type'] in ['directed', 'undirected']
```

### Round-Trip Validation
```python
# Implemented in src/6_validation.py
def validate_round_trip(original_gnn, exported_formats):
    # Parse original
    parsed = parse_gnn(original_gnn)
    
    # Export and re-import each format
    for format_name, format_data in exported_formats.items():
        reimported = import_format(format_data, format_name)
        assert semantic_equivalent(parsed, reimported)
```

This schema forms the foundation for all pipeline processing.
