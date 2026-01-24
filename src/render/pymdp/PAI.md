# PyMDP Renderer - PAI Context

## Quick Reference

**Purpose:** Render POMDP models to executable PyMDP Python code.

**When to use this module:**
- Generate PyMDP agent code from POMDPStateSpace
- Create simulation scripts for Python execution

## Common Operations

```python
from render.pymdp.pymdp_renderer import PyMDPRenderer
renderer = PyMDPRenderer()
code = renderer.render(pomdp_state_space)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace |
| **Output** | execute/pymdp | *_pymdp.py |

## Tips for AI Assistants

1. **Framework:** PyMDP (Python Active Inference)
2. **Templates:** Uses Jinja2 templates
3. **Matrices:** Converts A, B, C, D to numpy arrays

---

**Version:** 1.1.3 | **Step:** 11 (Render)
