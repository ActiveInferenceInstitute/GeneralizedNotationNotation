# GNN Architecture Reference

Implementation details of the thin orchestrator pattern and cross-module integration.

## Thin Orchestrator Pattern (Actual Implementation)

### Pattern Definition
Each numbered pipeline step (0-23) follows this structure:
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
from visualization import process_visualization_main  # ← Core implementation

run_script = create_standardized_pipeline_script(
    "8_visualization.py",
    process_visualization_main,  # ← Delegates to module
    "Matrix and network visualization processing"
)

def main() -> int:
    return run_script()  # ← Pure orchestration
```

#### Module Implementation (src/visualization/__init__.py)
```python
# Exposes core functionality with safe imports
try:
    from .visualizer import GNNVisualizer, generate_graph_visualization
    from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
    from .processor import generate_matrix_visualizations, generate_network_visualizations
except Exception:
    # Fallback implementations for missing dependencies
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
├── actinf_pomdp_agent_pymdp.py      (Generated PyMDP code)
├── actinf_pomdp_agent_rxinfer.jl    (Generated RxInfer code) 
└── render_summary.json              (Generation metadata)

src/12_execute.py:
├── Discovers generated files in output/11_render_output/
├── Executes: subprocess.run(["python", "actinf_pomdp_agent_pymdp.py"])
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
├── analysis_utils.py                  # Core functions (line 13-62)
│   ├── analyze_variable_types() → Dict[str, Any]
│   ├── analyze_connections() → Dict[str, Any]
│   └── estimate_computational_complexity() → Dict[str, Any]
├── checker.py                         # GNNTypeChecker (line 174)
│   ├── check_file() → Tuple[bool, List[str], List[str], Dict[str, Any]]
│   ├── _check_required_sections()
│   └── _collect_variable_analysis()
└── processor.py                       # GNNTypeChecker (line 20)
    ├── _validate_type() (line 191)
    └── _analyze_types() (line 232)
```

### Visualization Modules (src/visualization/)
```
Visualization implementation hierarchy:

src/visualization/
├── __init__.py                        # Safe imports (line 15-47)
├── visualizer.py                      # GNNVisualizer (line 66)
├── matrix_visualizer.py               # MatrixVisualizer (line 40)
│   └── generate_matrix_visualizations() (line 941)
└── processor.py                       # Core processing functions
    ├── parse_matrix_data() (line 367)
    ├── generate_matrix_visualizations() (line 403)
    └── generate_network_visualizations() (line 523)
```

## Framework Integration Points (Implementation Details)

### PyMDP Integration
**Location:** `src/render/pymdp/` (to be implemented)
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
**Location:** `src/render/rxinfer/` (to be implemented)
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
**Location:** `src/render/discopy/` (to be implemented)  
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
        morphism = Arrow(source_obj, target_obj, name=f"{conn['source']}_to_{conn['target']}")
        diagram = diagram >> morphism
```

## Pipeline Orchestration Details

### Argument Flow (src/main.py → steps)
```python
# Main pipeline argument passing:
def execute_pipeline_step(script_name: str, args: PipelineArguments, logger):
    cmd = build_step_command_args(
        script_name.replace('.py', ''),
        args,  # Contains: target_dir, output_dir, verbose, etc.
        python_executable,
        script_path
    )
    
# Each step receives standardized arguments:
# --target-dir, --output-dir, --verbose, --recursive, [step-specific args]
```

### Output Directory Management
```python
# Centralized output directory structure:
# src/pipeline/config.py:get_output_dir_for_script()

def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    step_name = script_name.replace('.py', '')
    return base_output_dir / f"{step_name}_output"

# Results in structure:
# output/
# ├── 3_gnn_output/
# ├── 5_type_checker_output/  
# ├── 8_visualization_output/
# └── 11_render_output/
```

### Error Handling Pattern
```python
# Standardized error handling in each step:
try:
    success = module_processing_function(target_dir, output_dir, logger, **kwargs)
    return 0 if success else 1
except Exception as e:
    log_step_error(logger, f"Step processing failed: {e}")
    return 1  # Pipeline continues with next step
```

## Dependency Resolution (Actual Implementation)

### Safe Import Pattern (src/visualization/__init__.py:15-47)
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
    return create_fallback_html_report()
```

### MCP Integration Pattern 
Each module includes `mcp.py` with tool registration:
```python
# Example: src/visualization/mcp.py
@server.tool()
def visualize_gnn_model(content: str, output_path: str) -> dict:
    """Generate visualization for GNN model content."""
    return process_visualization_main(
        target_dir=Path(content),
        output_dir=Path(output_path)
    )
```

This architecture enables modular development, safe-to-fail operation, and framework interoperability while maintaining clear separation between orchestration and implementation logic.
