# ActiveInference.jl Executor - PAI Context

## Quick Reference

**Purpose:** Execute ActiveInference.jl Julia simulations and capture results.

**When to use this module:**
- Run ActiveInference.jl agent simulations
- Capture belief trajectories from CSV
- Generate analysis inputs

## Common Operations

```python
from execute.activeinference_jl.executor import ActiveInferenceJLExecutor
executor = ActiveInferenceJLExecutor()
results = executor.execute(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render/activeinference_jl | *_activeinference.jl |
| **Output** | analysis/activeinference_jl | simulation_results.csv |

## Tips for AI Assistants

1. **Runtime:** Julia with ActiveInference.jl package
2. **Subprocess:** Executes via julia command
3. **Results:** CSV with step, observation, action, beliefs

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
