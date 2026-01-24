# JAX Renderer - PAI Context

## Quick Reference

**Purpose:** Render POMDP models to executable JAX Python code with autodiff.

**When to use this module:**
- Generate JAX agent code from POMDPStateSpace
- Create differentiable simulations for Python execution

## Common Operations

```python
from render.jax.jax_renderer import JAXRenderer
renderer = JAXRenderer()
code = renderer.render(pomdp_state_space)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace |
| **Output** | execute/jax | *_jax.py |

## Tips for AI Assistants

1. **Framework:** JAX (differentiable Python)
2. **Templates:** Uses Jinja2 templates
3. **Autodiff:** Supports gradient computation

---

**Version:** 1.1.3 | **Step:** 11 (Render)
