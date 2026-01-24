# RxInfer Renderer - PAI Context

## Quick Reference

**Purpose:** Render POMDP models to executable RxInfer.jl Julia code.

**When to use this module:**
- Generate RxInfer.jl agent code from POMDPStateSpace
- Create message-passing simulations for Julia execution

## Common Operations

```python
from render.rxinfer.rxinfer_renderer import RxInferRenderer
renderer = RxInferRenderer()
code = renderer.render(pomdp_state_space)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace |
| **Output** | execute/rxinfer | *_rxinfer.jl |

## Tips for AI Assistants

1. **Framework:** RxInfer.jl (Julia reactive inference)
2. **Templates:** Uses Jinja2 templates
3. **Factor Graphs:** Generates factor graph specifications

---

**Version:** 1.1.3 | **Step:** 11 (Render)
