# GNN Rendering Subsystem

This directory contains the code responsible for rendering Generalized Notation Notation (GNN) specifications into executable formats for various target platforms.

## Structure Overview

The rendering subsystem is organized into modular components:

### Core Components
- `render.py` - Main entry point that coordinates rendering to different target platforms
- `mcp.py` - Model Context Protocol integration for renderer tools

### PyMDP Renderer (Modular Implementation)
- `pymdp_renderer.py` - Main entry point for PyMDP rendering, coordinates the rendering process
- `pymdp_converter.py` - Core conversion logic from GNN to PyMDP format
- `pymdp_templates.py` - Templates for generating Python code from GNN specifications
- `pymdp_utils.py` - Utility functions for matrix manipulation and code generation
- `pymdp.py` - Legacy implementation (deprecated)

### RxInfer Renderer (Planned Modularization)
- `rxinfer.py` - Current monolithic implementation (to be modularized)
- `rxinfer_renderer.py` - Main entry point for RxInfer rendering (planned)
- `rxinfer_converter.py` - Core conversion logic from GNN to RxInfer format (planned)
- `rxinfer_templates.py` - Templates for generating Julia code from GNN specifications (planned)
- `rxinfer_utils.py` - Utility functions for Julia code generation (planned)

## Usage

The render subsystem is typically invoked via the main pipeline in `src/main.py`, but can also be used directly:

```python
from render.render import render_gnn_spec
import json
from pathlib import Path

# Load a GNN specification
with open("path/to/gnn_spec.json", "r") as f:
    gnn_spec = json.load(f)

# Render to PyMDP
success, message, artifacts = render_gnn_spec(
    gnn_spec,
    Path("output/pymdp_script.py"),
    "pymdp",
    {"include_example_usage": True}
)

# Render to RxInfer
success, message, artifacts = render_gnn_spec(
    gnn_spec,
    Path("output/rxinfer_script.jl"),
    "rxinfer",
    {"include_inference_script": True}
)
```

## Modularization Plan

### Completed
- [x] Separated PyMDP renderer into modular components
- [x] Implemented GnnToPyMdpConverter class
- [x] Created template generation functions
- [x] Added utility functions for code generation
- [x] Completed modular implementation of PyMDP matrix conversion methods
- [x] Added deprecation warnings to legacy implementations
- [x] Updated documentation

### In Progress
- [ ] Add unit tests for PyMDP components

### Planned
- [ ] Modularize RxInfer renderer following similar pattern
- [ ] Create unit tests for RxInfer components
- [ ] Create developer guide for extending to new targets

## Contributing

When adding support for new target platforms, follow the modular design pattern:
1. Create a converter class that transforms GNN data into target-specific structures
2. Implement template functions for code generation
3. Add utility functions for platform-specific tasks
4. Create a renderer module that coordinates the process
5. Update the main `render.py` to include the new target 

## Output Structure

The rendering process generates files under `output/gnn_rendered_simulators/` with subfolders for each framework:

- `pymdp/`: Python scripts for PyMDP simulations (e.g., model_name_pymdp.py)
- `rxinfer/`: Julia scripts for RxInfer models (e.g., model_name_rxinfer.jl)
- `discopy/`: Python scripts for DisCoPy diagrams (e.g., model_name_discopy.py)
- `activeinference_jl/`: Julia scripts for ActiveInference.jl (e.g., model_name_activeinference.jl)
- `jax/`: Python scripts for JAX models (e.g., model_name_jax.py)

Each script is a self-contained executable that can be run to simulate or evaluate the model.

## Running Generated Scripts

### PyMDP
python model_name_pymdp.py

### RxInfer
julia model_name_rxinfer.jl

### DisCoPy
python model_name_discopy.py

### ActiveInference.jl
julia model_name_activeinference.jl

### JAX
python model_name_jax.py

## Extending to New Backends

To add a new rendering backend:
1. Create a subfolder under src/render/ (e.g., new_backend/)
2. Implement a render_gnn_to_newbackend function that takes gnn_spec dict and output_path, generates the script.
3. Add to __init__.py import and availability check.
4. Update render.py render_gnn_spec to handle the new target.
5. Add to 9_render.py render_targets list.

See existing backends for patterns.

## Best Practices

- Ensure parsers handle all GNN sections relevant to your backend.
- Generate self-contained scripts with imports and main execution.
- Include comments and documentation in generated code.
- Handle errors gracefully and log issues.
- Test with sample GNN files. 