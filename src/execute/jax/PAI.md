# JAX Executor - PAI Context

## Quick Reference

**Purpose:** Execute JAX Python simulations with autodiff and capture results.

**When to use this module:**
- Run JAX differentiable agent simulations
- Capture gradients and convergence
- Generate simulation outputs

## Common Operations

```python
from execute.jax.jax_runner import run_jax_simulation
results = run_jax_simulation(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render/jax | *_jax.py |
| **Output** | analysis/jax | jax_outputs/*.json |

## Tips for AI Assistants

1. **Runtime:** Python with JAX package
2. **Subprocess:** Executes in isolated process
3. **Results:** JSON with gradients, losses, beliefs

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
