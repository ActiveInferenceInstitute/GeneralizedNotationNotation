# GNN API Reference

> **📋 Document Metadata**  
> **Type**: API Reference | **Audience**: Developers & Integrators | **Complexity**: Intermediate-Advanced  
> **Last Updated**: June 2025 | **Status**: Production-Ready  
> **Cross-References**: [Pipeline Architecture](../pipeline/README.md) | [Framework Integration](../gnn/framework_integration_guide.md)

## Overview
This is the complete API reference for GeneralizedNotationNotation (GNN), covering all public interfaces, classes, and functions.

## Core Modules

### `src.gnn` - Core GNN Processing
Main module for GNN file parsing and processing.

#### Classes

##### `GNNModel`
**Description**: Represents a complete GNN model with all sections and metadata.

```python
class GNNModel:
    def __init__(self, filepath: str = None, content: str = None)
    
    # Properties
    model_name: str
    version: str
    processing_flags: List[str]
    state_space: Dict[str, Variable]
    connections: List[Connection]
    parameters: Dict[str, Any]
    equations: List[str]
    time_settings: TimeSettings
    
    # Methods
    def validate(self) -> ValidationResult
    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
    def save(self, filepath: str) -> None
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GNNModel'
    
    @classmethod
    def from_string(cls, content: str) -> 'GNNModel'
```

**Usage Example**:
```python
from src.gnn import GNNModel

# Load from file
model = GNNModel.from_file("examples/basic_model.md")

# Validate syntax
result = model.validate()
if result.is_valid:
    print("Model is valid!")
else:
    print(f"Errors: {result.errors}")

# Export to JSON
json_output = model.to_json()
```

##### `Variable`
**Description**: Represents a GNN variable with dimensions and type information.

```python
class Variable:
    def __init__(self, name: str, dimensions: List[int], 
                 var_type: str = "categorical")
    
    name: str
    dimensions: List[int]
    var_type: str  # "categorical", "continuous", "binary"
    description: str
    
    def total_size(self) -> int
    def is_compatible_with(self, other: 'Variable') -> bool
```

##### `Connection`
**Description**: Represents relationships between variables.

```python
class Connection:
    def __init__(self, source: str, target: str, 
                 connection_type: str = "directed")
    
    source: str
    target: str
    connection_type: str  # "directed", "undirected"
    strength: float = 1.0
    
    def is_directed(self) -> bool
    def reverse(self) -> 'Connection'
```

#### Functions

##### `gnn_parse_file(filepath: str) -> GNNModel`
**Description**: Parse a GNN file and return a model object.

**Parameters**:
- `filepath` (str): Path to the .md GNN file

**Returns**: 
- `GNNModel`: Parsed model object

**Raises**:
- `GNNSyntaxError`: If file contains syntax errors
- `FileNotFoundError`: If file doesn't exist

**Example**:
```python
from src.gnn import gnn_parse_file

try:
    model = gnn_parse_file("my_model.md")
    print(f"Loaded model: {model.model_name}")
except GNNSyntaxError as e:
    print(f"Syntax error: {e}")
```

##### `gnn_validate_syntax(model: GNNModel) -> ValidationResult`
**Description**: Validate GNN model syntax and structure.

### `src.type_checker` - Model Validation

#### Classes

##### `TypeChecker`
**Description**: Validates GNN models for type consistency and correctness.

```python
class TypeChecker:
    def __init__(self, strict_mode: bool = True)
    
    def check_model(self, model: GNNModel) -> ValidationResult
    def check_dimensions(self, variables: Dict[str, Variable]) -> List[Error]
    def check_connections(self, connections: List[Connection], 
                         variables: Dict[str, Variable]) -> List[Error]
    def estimate_resources(self, model: GNNModel) -> ResourceEstimate
```

##### `ValidationResult`
**Description**: Result of model validation with errors and warnings.

```python
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    
    def summary(self) -> str
    def to_dict(self) -> Dict[str, Any]
```

### `src.export` - Export Functionality

#### Functions

##### `export_to_json(model: GNNModel, filepath: str) -> None`
**Description**: Export GNN model to JSON format.

##### `export_to_xml(model: GNNModel, filepath: str) -> None`
**Description**: Export GNN model to XML format.

##### `export_to_graphml(model: GNNModel, filepath: str) -> None`
**Description**: Export GNN model to GraphML for network analysis.

##### `export_to_dot(model: GNNModel, filepath: str) -> None`
**Description**: Export GNN model to Graphviz DOT format.

**Example**:
```python
from src.export import export_to_json, export_to_xml
from src.gnn import GNNModel

model = GNNModel.from_file("example.md")
export_to_json(model, "output.json")
export_to_xml(model, "output.xml")
```

### `src.visualization` - Visualization Tools

#### Classes

##### `GNNVisualizer`
**Description**: Creates visual representations of GNN models.

```python
class GNNVisualizer:
    def __init__(self, layout: str = "spring", 
                 format: str = "png")
    
    def visualize_model(self, model: GNNModel, 
                       output_path: str) -> None
    def visualize_connections(self, connections: List[Connection],
                            output_path: str) -> None
    def create_dependency_graph(self, model: GNNModel) -> nx.Graph
```

**Usage**:
```python
from src.visualization import GNNVisualizer
from src.gnn import GNNModel

model = GNNModel.from_file("example.md")
viz = GNNVisualizer(layout="hierarchical", format="svg")
viz.visualize_model(model, "model_graph.svg")
```

### `src.render` - Code Generation

#### PyMDP Backend

##### `PyMDPRenderer`
**Description**: Generates PyMDP simulation code from GNN models.

```python
class PyMDPRenderer:
    def __init__(self, output_dir: str = "rendered_pymdp/")
    
    def render_model(self, model: GNNModel) -> str
    def generate_agent_script(self, model: GNNModel) -> str
    def create_simulation_config(self, model: GNNModel) -> Dict[str, Any]
```

#### RxInfer Backend

##### `RxInferRenderer`
**Description**: Generates RxInfer.jl simulation code from GNN models.

```python
class RxInferRenderer:
    def __init__(self, output_dir: str = "rendered_rxinfer/")
    
    def render_model(self, model: GNNModel) -> str
    def generate_julia_script(self, model: GNNModel) -> str
    def create_toml_config(self, model: GNNModel) -> str
```

### `src.execute` - Simulation Execution

#### Classes

##### `SimulationRunner`
**Description**: Executes rendered simulation scripts.

```python
class SimulationRunner:
    def __init__(self, backend: str = "pymdp")
    
    def run_simulation(self, script_path: str, 
                      config: Dict[str, Any] = None) -> SimulationResult
    def batch_run(self, script_paths: List[str]) -> List[SimulationResult]
```

### `src.llm` - LLM Integration

#### Classes

##### `LLMAnalyzer`
**Description**: Uses LLMs to analyze and enhance GNN models.

```python
class LLMAnalyzer:
    def __init__(self, model_name: str = "gpt-4", 
                 api_key: str = None)
    
    def analyze_model(self, model: GNNModel) -> LLMAnalysis
    def suggest_improvements(self, model: GNNModel) -> List[Suggestion]
    def explain_model(self, model: GNNModel) -> str
    def validate_with_llm(self, model: GNNModel) -> LLMValidation
```

### `src.discopy_translator_module` - Categorical Diagrams

#### Classes

##### `DisCoPyTranslator`
**Description**: Translates GNN models to DisCoPy categorical diagrams.

```python
class DisCoPyTranslator:
    def __init__(self, category_type: str = "pregroup")
    
    def translate_model(self, model: GNNModel) -> discopy.Diagram
    def create_string_diagram(self, model: GNNModel) -> discopy.Diagram
    def compose_diagrams(self, diagrams: List[discopy.Diagram]) -> discopy.Diagram
```

### `src.mcp` - Model Context Protocol

#### Classes

##### `MCPServer`
**Description**: MCP server for AI assistant integration.

```python
class MCPServer:
    def __init__(self, port: int = 8000)
    
    def register_tools(self, tools: List[MCPTool]) -> None
    def start_server(self) -> None
    def handle_request(self, request: MCPRequest) -> MCPResponse
```

## Utility Modules

### `src.utils` - Shared Utilities

#### Functions

##### `load_config(config_path: str) -> Dict[str, Any]`
**Description**: Load configuration from YAML/JSON file.

##### `setup_logging(level: str = "INFO", log_file: str = None) -> None`
**Description**: Configure logging for GNN pipeline.

##### `create_output_directory(base_dir: str, timestamp: bool = True) -> str`
**Description**: Create timestamped output directory.

## Error Handling

### Exception Classes

```python
class GNNError(Exception):
    """Base exception for GNN-related errors"""
    pass

class GNNSyntaxError(GNNError):
    """Raised when GNN file has syntax errors"""
    def __init__(self, message: str, line_number: int = None,
                 section: str = None):
        self.line_number = line_number
        self.section = section
        super().__init__(message)

class GNNValidationError(GNNError):
    """Raised when model validation fails"""
    pass

class GNNRenderError(GNNError):
    """Raised when code generation fails"""
    pass

class GNNExecutionError(GNNError):
    """Raised when simulation execution fails"""
    pass
```

## Type Definitions

### Common Types

```python
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum

class VariableType(Enum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    BINARY = "binary"

class ConnectionType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"

@dataclass
class TimeSettings:
    is_dynamic: bool
    time_type: str  # "DiscreteTime", "ContinuousTime"
    horizon: Optional[int] = None

@dataclass
class ResourceEstimate:
    memory_mb: float
    compute_complexity: str
    estimated_runtime_seconds: float
```

## Configuration

### Pipeline Configuration

```python
# config.yaml structure
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  parallel: true
  output_dir: "output/"
  cleanup: false

gnn:
  validation:
    strict_mode: true
    check_dimensions: true
    check_stochasticity: true
  
export:
  formats: ["json", "xml", "graphml"]
  pretty_print: true

visualization:
  layout: "spring"
  format: "png"
  dpi: 300

llm:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 2000
```

## Examples

### Complete Workflow Example

```python
import os
from src.gnn import GNNModel
from src.type_checker import TypeChecker
from src.export import export_to_json
from src.visualization import GNNVisualizer
from src.render.pymdp import PyMDPRenderer
from src.execute import SimulationRunner

# Load and validate model
model = GNNModel.from_file("examples/basic_model.md")
checker = TypeChecker(strict_mode=True)
result = checker.check_model(model)

if result.is_valid:
    # Export model
    export_to_json(model, "output/model.json")
    
    # Visualize
    viz = GNNVisualizer()
    viz.visualize_model(model, "output/model_graph.png")
    
    # Render simulation
    renderer = PyMDPRenderer()
    script = renderer.render_model(model)
    
    with open("output/simulation.py", "w") as f:
        f.write(script)
    
    # Execute
    runner = SimulationRunner(backend="pymdp")
    sim_result = runner.run_simulation("output/simulation.py")
    
    print(f"Simulation completed: {sim_result.success}")
else:
    print(f"Model validation failed: {result.errors}")
```

This API reference provides the complete interface for working with GNN programmatically. For more examples, see the tutorials and pipeline documentation. 