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
# Render to all frameworks
from render.processor import RenderProcessor
processor = RenderProcessor(input_dir, output_dir)
results = processor.process(verbose=True)

# Render to specific framework
from render.pymdp.pymdp_renderer import PyMDPRenderer
renderer = PyMDPRenderer()
code = renderer.render(pomdp_state_space)
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
2. **Templates:** Each renderer uses Jinja2 templates for code generation
3. **Output Location:** `output/11_render_output/{model}/{framework}/`
4. **Matrix Handling:** A, B, C, D matrices are converted to framework-native formats

---

**Version:** 1.1.3 | **Step:** 11 (Render)
