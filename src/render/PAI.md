# Render Module - PAI Context

## Quick Reference

**Purpose:** Transform parsed GNN/POMDP models into executable framework-specific code.

**When to use this module:**
- Generate PyMDP Python code from POMDP
- Generate RxInfer.jl Julia code from POMDP
- Generate ActiveInference.jl Julia code from POMDP
- Generate JAX Python code from POMDP
- Generate DisCoPy Python code from POMDP

## Common Operations

```python
from pathlib import Path
from render import process_render, render_gnn_spec

# Render all models in a directory (Step 11 entrypoint)
ok = process_render(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/11_render_output"),
    verbose=True,
)

# Render a single parsed spec to one target
success, msg, artifacts = render_gnn_spec(
    gnn_spec=parsed_spec,
    target="pymdp",
    output_directory=Path("output/11_render_output/single/pymdp"),
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace objects |
| **Output** | execute | Framework-specific code files |

## Framework Renderers

| Framework | Renderer | Output |
|-----------|----------|--------|
| PyMDP | `pymdp/pymdp_renderer.py` | `*_pymdp.py` |
| RxInfer.jl | `rxinfer/rxinfer_renderer.py` | `*_rxinfer.jl` |
| ActiveInference.jl | `activeinference_jl/activeinference_renderer.py` | `*_activeinference.jl` |
| JAX | `jax/jax_renderer.py` | `*_jax.py` |
| DisCoPy | `discopy/discopy_renderer.py` | `*_discopy.py` |

## Key Files

- `processor.py` - Orchestrates all renderers
- `pomdp_processor.py` - POMDP-specific processing
- `{framework}/{framework}_renderer.py` - Framework implementations

## Tips for AI Assistants

1. **Step 11:** Render is the bridge between parsing and execution
2. **Templates:** Code generation uses Python string templates / f-strings (not Jinja2)
3. **Output Location:** `output/11_render_output/{model}/{framework}/`
4. **Matrix Handling:** A, B, C, D matrices are converted to framework-native formats

---

**Version:** 1.1.3 | **Step:** 11 (Render)
