# GNN Schema Specification

**Version**: v3.2.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: GNN parser and schema behavior. See [framework implementations](../implementations/README.md) for current backend coverage.

Complete specification for GNN syntax parsing and validation.

## Pipeline Processing

GNN schema validation is handled by multiple pipeline steps:

- **`src/3_gnn.py`** → GNN file parsing and schema validation
  - Implementation: `src/gnn/schema_validator.py`
  - See: **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**
- **`src/5_type_checker.py`** → Type and dimensional validation
  - Implementation: `src/type_checker/checking/core.py` (`GNNTypeChecker`)
  - See: **[src/type_checker/AGENTS.md](../../../src/type_checker/AGENTS.md)**
- **`src/6_validation.py`** → Advanced consistency checking
  - See: **[src/validation/AGENTS.md](../../../src/validation/AGENTS.md)**

**Quick Start:**

```bash
# Validate GNN schema
python src/main.py --only-steps "3,5,6" --target-dir input/gnn_files --verbose
```

For complete pipeline documentation, see **[src/AGENTS.md](../../../src/AGENTS.md)**.

---

## Core Schema Components

### Variable Declaration

```gnn
<name>[<dimensions>,type=<type>]
```

**Implementation:** `src/gnn/schema.py:parse_state_space()` (strict) / `src/gnn/schema_validator.py:VARIABLE_PATTERN` (permissive)

**Schema Rules:**

- `<name>`: `[a-zA-Z_][a-zA-Z0-9_]*` (identifier pattern)
- `<dimensions>`: comma-separated dimension tokens. Usually integers, but the
  permissive pattern accepts `[^\]]+`, so named references are legal —
  `G[pi,type=float]` appears in `input/gnn_files/discrete/tmaze_epistemic.md`
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

**Implementation:** `src/gnn/schema.py:parse_connections()` (strict) / `src/gnn/schema_validator.py:CONNECTION_PATTERN` (permissive)

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

**Implementation:** `src/gnn/schema.py:validate_required_sections()`

**Required sections** — the exact contents of
`src/gnn/schema.py::REQUIRED_SECTIONS`. A missing one is a hard `GNN-E001`
error:

- `## GNNSection` - Short, space-free model identifier
- `## GNNVersionAndFlags` - Specification version and optional flags
- `## ModelName` - Human-readable model title
- `## StateSpaceBlock` - Variable declarations
- `## Connections` - Connection specifications

**Not enforced by the validator**, but expected by downstream steps — a model
without `InitialParameterization` cannot render, and one without
`ActInfOntologyAnnotation` is invisible to Steps 10, 13, and 24:

- `## ModelAnnotation` - Description
- `## InitialParameterization` - Parameter values
- `## Equations` - Mathematical relations
- `## Time` - Temporal regime and horizon
- `## ActInfOntologyAnnotation` - Semantic mappings
- `## ModelParameters` - Scalar parameters read by renderers
- `## Footer` / `## Signature` - Closure and provenance

The full obligation table is in
[`gnn_syntax.md`](gnn_syntax.md#canonical-section-inventory).

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

**Entry Point:** `src/5_type_checker.py:main()` → `src/type_checker/analysis_utils.py:analyze_variable_types()`
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
- **ActiveInference.jl**: `src/render/activeinference_jl/` → `.jl` files
- **DisCoPy**: `src/render/discopy/` → categorical diagrams

## Core Method Locations (Actual Implementation)

### Parsing Pipeline (Step 3: GNN Processing)

```text
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

```text
src/5_type_checker.py (thin orchestrator)
└── src/type_checker/
    ├── analysis_utils.py                 # standalone helpers, no classes
    │   ├── analyze_variable_types() (line 13)
    │   ├── analyze_connections() (line 78)
    │   └── estimate_computational_complexity() (line 131)
    ├── checking/
    │   ├── core.py
    │   │   └── GNNTypeChecker (line 111)
    │   │       ├── check_file() (line 118)
    │   │       ├── validate_gnn_files() (line 163)
    │   │       └── _analyze_types() (line 320)
    │   ├── dimensions.py
    │   └── rules.py
    ├── estimation/                       # GNNResourceEstimator
    └── processor.py                      # thin re-export facade (17 lines)
```

### Visualization Pipeline (Steps 8 & 9)

```
src/8_visualization.py (thin orchestrator)
└── src/visualization/
    ├── visualizer.py
    │   └── GNNVisualizer (line 61)
    ├── matrix/
    │   ├── visualizer.py
    │   │   ├── MatrixVisualizer (line 171)
    │   │   └── generate_matrix_visualizations() (line 1649)
    │   └── compat.py
    │       └── parse_matrix_data() (line 18)
    ├── graph/
    │   └── network_visualizations.py
    │       └── generate_network_visualizations() (line 82)
    ├── matrix_visualizer.py               # re-export facade (15 lines)
    ├── processor.py                       # re-export facade (34 lines)
    └── __init__.py                        # safe imports with alternatives

src/9_advanced_viz.py (thin orchestrator)
└── src/advanced_visualization/
    └── visualizer.py
        └── AdvancedVisualizer (line 38)
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

1. **Lexical / syntactic**: `src/gnn/schema.py` - section, declaration, and
   connection parsing (there is no separate lexer module)
2. **Structural**: `src/gnn/parser.py` and `src/gnn/parsers/` - multi-format
   parsing into the shared model dict
3. **Semantic**: `src/type_checker/checking/core.py` - type and dimension validation
4. **Ontological**: `src/ontology/processor.py` - domain validation

### Framework Integration Points

- **PyMDP**: Matrices → `pymdp.Agent(A=A, B=B, C=C, D=D)`
- **RxInfer.jl**: Probabilistic → `@model function gnn_model()`
- **DisCoPy**: Categories → `Diagram` objects with morphisms
- **JAX**: Arrays → `jax.numpy` optimized computations

## Validation Schema

### Variable Validation

```python
# Illustrative pseudocode — not a real symbol
def validate_variable(var):
    assert var["name"].isidentifier()
    assert all(d > 0 for d in var["dimensions"])
    assert var["type"] in ["int", "float", "double", "bool"]
```

### Connection Validation  

```python
# Illustrative pseudocode — not a real symbol
def validate_connection(conn, variables):
    assert conn["source"] in [v["name"] for v in variables]
    assert conn["target"] in [v["name"] for v in variables]
    assert conn["type"] in ["directed", "undirected"]
```

### Round-Trip Validation

```python
# Implemented in src/6_validation.py
# Illustrative pseudocode — not a real symbol
def validate_round_trip(original_gnn, exported_formats):
    # Parse original
    parsed = parse_gnn(original_gnn)
    
    # Export and re-import each format
    for format_name, format_data in exported_formats.items():
        reimported = import_format(format_data, format_name)
        assert semantic_equivalent(parsed, reimported)
```

This schema forms the foundation for all pipeline processing.
