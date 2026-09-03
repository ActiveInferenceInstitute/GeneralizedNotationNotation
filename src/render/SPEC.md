# Render Module Specification

## Overview
Multi-framework rendering and code generation from GNN models.

## Components

### Framework Renderers
- `jax/jax_renderer.py` - JAX code generation
- `pymdp/pymdp_renderer.py` - PyMDP runner generation
- `discopy/translator.py` - DisCoPy translation
- `rxinfer/rxinfer_renderer.py` - Canonical RxInfer.jl renderer (genuine `@model` + `infer()`); `toml_generator.py` is the retired TOML emitter (the `rxinfer_toml` target is no longer supported)
- `stan/stan_renderer.py` - Stan program + cmdstanpy driver generation
- `pytorch/pytorch_renderer.py`, `numpyro/numpyro_renderer.py` - generator-backed continuous backends

### Core
- `generators.py` - Code generator utilities
- `processor.py` - Step 11 entry point
- `framework_registry.py` - Canonical framework inventory (`supports_continuous`, availability)

## Supported Frameworks
- `pymdp`, `rxinfer`, `activeinference_jl`, `jax`, `discopy`, `pytorch`, `numpyro`, `stan`, and `bnlearn`.

## Key Exports
```python
from render import process_render, JAXRenderer
```

The authoritative export surface is `src/render/__init__.py`.


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
