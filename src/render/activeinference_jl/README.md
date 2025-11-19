# ActiveInference.jl Rendering Module

This submodule handles code generation specifically for ActiveInference.jl simulations.

## Overview

ActiveInference.jl is a Julia implementation of Active Inference providing comprehensive support for hierarchical planning and temporal dynamics. This module generates complete, executable ActiveInference.jl simulation code from GNN specifications.

## Module Structure

```
src/render/activeinference_jl/
├── __init__.py                         # Module initialization
├── README.md                           # This documentation
├── AGENTS.md                           # Detailed agent scaffolding
├── activeinference_renderer.py         # Main renderer
├── activeinference_renderer_fixed.py   # Fixed renderer version
└── activeinference_renderer_simple.py  # Simplified renderer
```

## Core Components

### ActiveInference Renderer (`activeinference_renderer.py`)

**Purpose**: Generate executable ActiveInference.jl code from GNN models

**Key Functions**:
- `generate_activeinference_code()` - Main code generation
- `render_agent()` - Agent configuration
- `render_hierarchical_structure()` - Hierarchical planning support
- `generate_inference_loop()` - Inference loop generation

### Simplified Renderer (`activeinference_renderer_simple.py`)

**Purpose**: Provide simpler rendering for basic models

**Features**:
- Minimal dependencies
- Faster generation
- Suitable for simple models

## Features

- Full hierarchical Active Inference agent support
- Temporal dynamics handling
- Multi-level planning
- Julia optimization support
- Performance profiling integration

## Usage

```python
from render.activeinference_jl import generate_activeinference_jl_code

code = generate_activeinference_jl_code(
    model_data=parsed_gnn_model,
    output_path="output.jl"
)
```

## Output

Generated Julia code includes:
- Complete agent structure
- Hierarchical planning loops
- Inference implementation
- Result collection
- Performance metrics

## Dependencies

- `julia` - Julia runtime (optional, fallback: skip Julia generation)
- `ActiveInference.jl` - Julia package (must be installed in Julia)

## Testing

Comprehensive Julia tests ensure:
- Syntax correctness
- Execution success
- Output validation
- Performance characteristics

---

**Last Updated**: October 28, 2025  
**Status**: ✅ Production Ready


