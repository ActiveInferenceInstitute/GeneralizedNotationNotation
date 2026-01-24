# PyMDP Executor - PAI Context

## Quick Reference

**Purpose:** Execute PyMDP Python simulations and capture results.

**When to use this module:**
- Run PyMDP agent simulations
- Capture beliefs, actions, observations
- Generate simulation_results.json

## Common Operations

```python
from execute.pymdp.executor import PyMDPExecutor
executor = PyMDPExecutor()
results = executor.execute(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render/pymdp | *_pymdp.py |
| **Output** | analysis/pymdp | simulation_results.json |

## Tips for AI Assistants

1. **Runtime:** Python with pymdp package
2. **Subprocess:** Executes in isolated process
3. **Results:** JSON format with beliefs, actions

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
