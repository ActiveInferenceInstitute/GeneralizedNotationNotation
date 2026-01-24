# DisCoPy Executor - PAI Context

## Quick Reference

**Purpose:** Execute DisCoPy Python diagram simulations and capture results.

**When to use this module:**
- Run DisCoPy categorical diagram simulations
- Capture circuit information
- Generate diagram outputs

## Common Operations

```python
from execute.discopy.executor import DisCoPyExecutor
executor = DisCoPyExecutor()
results = executor.execute(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render/discopy | *_discopy.py |
| **Output** | analysis/discopy | circuit_info.json |

## Tips for AI Assistants

1. **Runtime:** Python with discopy package
2. **Subprocess:** Executes in isolated process
3. **Results:** JSON with diagram structure

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
