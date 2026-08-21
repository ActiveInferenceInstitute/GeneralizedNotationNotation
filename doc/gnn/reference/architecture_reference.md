# GNN Architecture Reference

**Version**: v3.0.0 Engine (Bundle v2.0.0)  
**Last Updated**: 2026-04-15  
**Status**: Maintained
**Scope**: Pipeline architecture and extension patterns. See [framework implementations](../implementations/README.md) for current backend coverage.

**GNN Architecture Team**  
**Version**: 2.0.0  
**Status**: Maintained
**Last Updated**: 2026-04-15  

Implementation details of the thin orchestrator pattern and cross-module integration.

For complete pipeline documentation:

- **[src/AGENTS.md](../../../src/AGENTS.md)**: Master agent scaffolding and module registry
- **[src/README.md](../../../src/README.md)**: Pipeline architecture and safety patterns
- **[src/main.py](../../../src/main.py)**: Pipeline orchestrator script

## Complete 25-Step Pipeline Mapping

The GNN pipeline consists of exactly 25 steps (0-24), each following the thin orchestrator pattern:

**Core Processing (Steps 0-9)**

- `0_template.py` → `src/template/` - Pipeline initialization
- `1_setup.py` → `src/setup/` - Environment and dependency setup
- `2_tests.py` → `src/tests/` - Test suite execution
- `3_gnn.py` → `src/gnn/` - GNN parsing and multi-format processing
- `4_model_registry.py` → `src/model_registry/` - Model versioning
- `5_type_checker.py` → `src/type_checker/` - Type validation
- `6_validation.py` → `src/validation/` - Consistency checking
- `7_export.py` → `src/export/` - Multi-format export
- `8_visualization.py` → `src/visualization/` - Graph visualization
- `9_advanced_viz.py` → `src/advanced_visualization/` - Advanced plots

**Simulation & Analysis (Steps 10-16)**

- `10_ontology.py` → `src/ontology/` - Ontology processing
- `11_render.py` → `src/render/` - Code generation
- `12_execute.py` → `src/execute/` - Simulation execution
- `13_llm.py` → `src/llm/` - LLM analysis
- `14_ml_integration.py` → `src/ml_integration/` - ML integration
- `15_audio.py` → `src/audio/` - Audio generation
- `16_analysis.py` → `src/analysis/` - Statistical analysis

**Integration & Output (Steps 17-24)**

- `17_integration.py` → `src/integration/` - System integration
- `18_security.py` → `src/security/` - Security validation
- `19_research.py` → `src/research/` - Research tools
- `20_website.py` → `src/website/` - Website generation
- `21_mcp.py` → `src/mcp/` - MCP processing
- `22_gui.py` → `src/gui/` - GUI interface
- `23_report.py` → `src/report/` - Report generation
- `24_intelligent_analysis.py` → `src/intelligent_analysis/` - AI-enhanced analysis

For module-specific documentation, see each `src/[module]/AGENTS.md` file.

## Thin Orchestrator Pattern (Actual Implementation)

### Pattern Definition

Each numbered pipeline step (0-24) follows this structure:

- **Thin orchestrator script** handles argument parsing, logging, output management
- **Module directory** contains actual implementation logic
- **Cross-references** between steps via standardized JSON outputs

### Pattern Implementation Example

#### Step 8: Visualization (src/8_visualization.py)

```python
#!/usr/bin/env python3
"""Step 8: Visualization Processing (Thin Orchestrator)"""

import sys
from pathlib import Path

# Thin orchestrator: delegates to module
from utils.pipeline_template import create_standardized_pipeline_script
from visualization import process_visualization  # ← Core implementation

run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization,  # ← Delegates to module
    "Matrix and network visualization processing",
)


def main() -> int:
    return run_script()  # ← Pure orchestration
```

#### Modular Implementation Layer

Example: `src/visualization/__init__.py`

```python
# Exposes core functionality with safe imports
try:
    from .visualizer import GNNVisualizer, generate_graph_visualization
    from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
    from .processor import (
        generate_matrix_visualizations,
        generate_network_visualizations,
    )
except Exception:
    # Alternative implementations for missing dependencies
    class GNNVisualizer:
        def __init__(self, *args, **kwargs):
            self.available = False

        def generate(self, *a, **k):
            return False
```

## Cross-Module Data Flow (Actual Files)

### Step 3 → Step 5: Parsed Data Transfer

```
Input:  input/gnn_files/actinf_pomdp_agent.md
Output: output/3_gnn_output/gnn_processing_results.json

Cross-reference in src/5_type_checker.py:
├── gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
├── gnn_results_file = gnn_nested_dir / "gnn_processing_results.json"  
└── with open(gnn_results_file, "r") as f: gnn_results = json.load(f)
```

### Step 5 → Step 8: Type Data Transfer  

```
Type data flows from Step 5 analysis to Step 8 visualization:

src/8_visualization.py:
└── visualizer.py:generate_matrix_visualization()
    ├── Reads: output/5_type_checker_output/type_check_results.json
    ├── Extracts: type_analysis["dimension_analysis"]
    └── Generates: matrix heatmaps based on dimensional analysis
```

### Step 11 → Step 12: Generated Code Execution

```  
Code generation to execution transfer:

src/11_render.py → output/11_render_output/
├── actinf_pomdp_agent_rxinfer.jl    (Generated RxInfer code) 
└── render_summary.json              (Generation metadata)

src/12_execute.py:
├── Discovers generated files in output/11_render_output/
└── Captures: execution results, timing, memory usage
```

## Module Structure Analysis (Real Locations)

### Parsing Modules (src/gnn/)

```
Core parsing functionality distribution:

src/gnn/
├── multi_format_processor.py          # Main processor (called by 3_gnn.py)
├── schema_validator.py                 # Regex patterns (line 58-63)
│   ├── SECTION_PATTERN
│   ├── VARIABLE_PATTERN  
│   ├── CONNECTION_PATTERN
│   └── PARAMETER_PATTERN
├── parser.py                          # GNNParsingSystem (line 72)
│   ├── _detect_format() (line 107)
│   └── _basic_parser() (line 120)
└── parsers/                           # Format-specific parsers
    ├── markdown_parser.py             # MarkdownGNNParser 
    ├── python_parser.py               # PythonGNNParser (line 25)
    ├── lean_parser.py                 # LeanGNNParser
    └── unified_parser.py              # UnifiedGNNParser
```

### Type Analysis Modules (src/type_checker/)

```
Type analysis implementation:

src/type_checker/
├── analysis_utils.py                  # Standalone analysis helpers
│   ├── analyze_variable_types() (line 13)
│   ├── analyze_connections() (line 78)
│   └── estimate_computational_complexity() (line 131)
├── checking/
│   ├── core.py                        # GNNTypeChecker (line 111)
│   │   ├── check_file() (line 118)
│   │   ├── validate_gnn_files() (line 163)
│   │   └── _analyze_types() (line 320)
│   ├── dimensions.py                  # Dimension consistency checks
│   └── rules.py                       # Rule definitions
├── estimation/                        # GNNResourceEstimator lives here
└── processor.py                       # Thin re-export facade (17 lines)
```

### Visualization Modules (src/visualization/)

```
Visualization implementation hierarchy:

src/visualization/
├── __init__.py                        # Safe imports
├── visualizer.py                      # GNNVisualizer (line 61)
├── matrix_visualizer.py               # Thin re-export facade (15 lines)
├── processor.py                       # Thin re-export facade (34 lines)
├── matrix/
│   ├── visualizer.py                  # MatrixVisualizer (line 171)
│   │   └── generate_matrix_visualizations() (line 1649)
│   └── compat.py                      # parse_matrix_data() (line 18)
└── graph/
    └── network_visualizations.py      # generate_network_visualizations() (line 82)
```

`matrix_visualizer.py` and `processor.py` are re-export facades kept for
import stability; the implementations moved into the `matrix/` and `graph/`
subpackages.

## Framework Integration Points (Implementation Details)

### PyMDP Integration

**Location:** `src/render/pymdp/`
**Template Variables:**

```python
# Matrix extraction from parsed GNN
A = extract_matrix(gnn_data, "A")  # Likelihood
B = extract_matrix(gnn_data, "B")  # Transitions
C = extract_vector(gnn_data, "C")  # Preferences
D = extract_vector(gnn_data, "D")  # Prior
E = extract_vector(gnn_data, "E")  # Habit

# PyMDP agent construction
agent = pymdp.Agent(A=A, B=B, C=C, D=D, E=E)
```

### RxInfer.jl Integration  

**Location:** `src/render/rxinfer/`
**Model Template:**

```julia
@model function gnn_model()
    # Extract GNN parameters
    A ~ MatrixDirichlet({{gnn_A_prior}})
    B ~ MatrixDirichlet({{gnn_B_prior}})
    
    # State evolution  
    s[1] ~ Categorical({{gnn_D}})
    for t in 2:T
        s[t] ~ Categorical(B[:, s[t-1], u[t-1]])
    end
    
    # Observation model
    for t in 1:T  
        o[t] ~ Categorical(A[:, s[t]])
    end
end
```

### DisCoPy Integration

**Location:** `src/render/discopy/`  
**Category Theory Mapping:**

```python
# GNN connections → DisCoPy morphisms
connections = gnn_data["connections"]
diagram = Id(X)  # Identity on objects

for conn in connections:
    if conn["type"] == "directed":
        # f: A → B becomes morphism
        source_obj = Object(conn["source"])
        target_obj = Object(conn["target"])
        morphism = Arrow(
            source_obj, target_obj, name=f"{conn['source']}_to_{conn['target']}"
        )
        diagram = diagram >> morphism
```

## Pipeline Orchestration Details

### Argument Flow (src/main.py → steps)

```python
# src/main.py → src/pipeline/execution.py
def execute_pipeline_step(script_name: str, args: PipelineArguments, logger):
    cmd = build_step_command_args(  # src/utils/argument_utils.py:1657
        script_name.replace(".py", ""),
        args,  # target_dir, output_dir, verbose, ...
        python_executable,
        script_path,
    )
```

Each step is invoked as a subprocess and receives the common flags
(`--target-dir`, `--output-dir`, `--verbose`) plus any step-specific arguments.

### Standardized I/O and State Management

Step scripts do not wire up logging or output paths by hand. They are built by
`create_standardized_pipeline_script` (`src/utils/pipeline_template.py`), which
supplies the logger and the resolved output directory:

```python
from utils.pipeline_template import create_standardized_pipeline_script

run = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN file discovery and multi-format parsing",
)
```

Output directories are resolved by
`get_output_dir_for_script` (`src/pipeline/config.py:136`), which maps each
script to a numbered sibling directory:

```text
output/
├── 3_gnn_output/
├── 5_type_checker_output/
├── 8_visualization_output/
└── 11_render_output/
```

### Error Handling Pattern

The template wrapper catches exceptions from the module function, logs them,
and converts the outcome into the pipeline exit-code contract
(`0` success, `1` error, `2` success with warnings):

```python
try:
    success = module_processing_function(target_dir, output_dir, logger, **kwargs)
except Exception as e:
    logger.error(f"Critical error in processing: {e}")
    return 1
return 0 if success else 1
```

## Dependency Resolution (Actual Implementation)

### Safe Import Pattern (src/visualization/**init**.py:15-47)

```python
# Pattern used across modules for optional dependencies:
try:
    from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
except Exception:
    MatrixVisualizer = None
    process_matrix_visualization = None

# Allows graceful degradation when dependencies missing
if MatrixVisualizer is None:
    logger.warning("Matrix visualization unavailable - matplotlib missing")
    return create_alternative_html_report()
```

### MCP Integration Pattern

Each module includes `mcp.py` with tool registration:

```python
# Example: src/visualization/mcp.py
@server.tool()
def visualize_gnn_model(content: str, output_path: str) -> dict:
    """Generate visualization for GNN model content."""
    return process_visualization(target_dir=Path(content), output_dir=Path(output_path))
```

This architecture enables modular development, safe-to-fail operation, and framework interoperability while maintaining clear separation between orchestration and implementation logic.
