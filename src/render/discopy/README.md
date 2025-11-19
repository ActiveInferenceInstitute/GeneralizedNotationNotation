# DisCoPy Rendering Module

This submodule handles code generation specifically for DisCoPy (Categorical Composition in Python) diagrams.

## Overview

DisCoPy is a Python library for computing with categorical structures including string diagrams and monoidal functors. This module generates DisCoPy code for visualizing and reasoning about compositional models from GNN specifications.

## Module Structure

```
src/render/discopy/
├── __init__.py                 # Module initialization
├── README.md                   # This documentation
├── AGENTS.md                   # Detailed agent scaffolding
├── discopy_renderer.py         # Main DisCoPy renderer
├── translator.py               # GNN to DisCoPy translator
└── visualize_jax_output.py     # Visualization utilities
```

## Core Components

### DisCoPy Renderer (`discopy_renderer.py`)

**Purpose**: Generate DisCoPy string diagrams from GNN models

**Key Functions**:
- `generate_discopy_code()` - Main code generation
- `render_diagram()` - Render categorical diagrams
- `compose_structures()` - Compose categorical structures
- `optimize_diagram()` - Optimize diagram representation

### GNN to DisCoPy Translator (`translator.py`)

**Purpose**: Translate GNN specifications to DisCoPy structures

**Features**:
- Automatic type inference
- Categorical relationship mapping
- Composition rule generation

## Features

- String diagram generation
- Categorical type system
- Compositional reasoning
- Automated diagram optimization
- Visual diagram rendering

## Usage

```python
from render.discopy import generate_discopy_code

code = generate_discopy_code(
    model_data=parsed_gnn_model,
    output_path="output.py"
)
```

## Output

Generated DisCoPy code includes:
- Type definitions
- Categorical structures
- String diagrams
- Composition operations
- Visualization code

## Dependencies

- `discopy` - DisCoPy package (optional, fallback: skip DisCoPy generation)
- `graphviz` - Graph visualization (optional)

## Testing

Tests ensure:
- Correct categorical structure
- Diagram composition validity
- Type consistency
- Visual rendering

---

**Last Updated**: October 28, 2025  
**Status**: ✅ Production Ready


