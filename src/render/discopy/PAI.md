# DisCoPy Renderer - PAI Context

## Quick Reference

**Purpose:** Render POMDP models to executable DisCoPy Python code with categorical diagrams.

**When to use this module:**
- Generate DisCoPy diagram code from POMDPStateSpace
- Create categorical quantum mechanics representations

## Common Operations

```python
from render.discopy.discopy_renderer import DisCoPyRenderer
renderer = DisCoPyRenderer()
code = renderer.render(pomdp_state_space)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | POMDPStateSpace |
| **Output** | execute/discopy | *_discopy.py |

## Tips for AI Assistants

1. **Framework:** DisCoPy (categorical diagrams)
2. **Templates:** Uses Jinja2 templates
3. **Diagrams:** Generates string diagram representations

---

**Version:** 1.1.3 | **Step:** 11 (Render)
