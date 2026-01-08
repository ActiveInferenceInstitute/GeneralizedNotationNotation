# Render Module Specification

## Overview
Multi-framework rendering and code generation from GNN models.

## Components

### Framework Renderers
- `jax/jax_renderer.py` - JAX code generation (1717 lines)
- `pymdp/pymdp_converter.py` - PyMDP integration (1517 lines)
- `discopy/translator.py` - DisCoPy translation (1684 lines)
- `rxinfer/toml_generator.py` - RxInfer TOML generation (1007 lines)

### Core
- `generators.py` - Code generator utilities (1365 lines)

## Supported Frameworks
- JAX, PyMDP, DisCoPy, RxInfer, NumPy

## Key Exports
```python
from render import process_render, JaxRenderer, PyMDPConverter
```
