# ActiveInference.jl Renderer - PAI Context

## Quick Reference

**Purpose:** Render POMDP models to executable ActiveInference.jl Julia code.

**When to use this module:**
- Generate ActiveInference.jl agent code from POMDPStateSpace
- Create simulations for Julia execution

## Common Operations

```python
from render.activeinference_jl.activeinference_renderer import ActiveInferenceRenderer
renderer = ActiveInferenceRenderer()
code = renderer.render(pomdp_state_space)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace |
| **Output** | execute/activeinference_jl | *_activeinference.jl |

## Tips for AI Assistants

1. **Framework:** ActiveInference.jl (Julia)
2. **Templates:** Uses Jinja2 templates
3. **Output:** CSV simulation results

---

**Version:** 1.1.3 | **Step:** 11 (Render)
