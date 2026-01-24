# RxInfer Executor - PAI Context

## Quick Reference

**Purpose:** Execute RxInfer.jl Julia simulations and capture results.

**When to use this module:**
- Run RxInfer.jl message-passing simulations
- Capture inference results and convergence
- Generate simulation_results.json

## Common Operations

```python
from execute.rxinfer.executor import RxInferExecutor
executor = RxInferExecutor()
results = executor.execute(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render/rxinfer | *_rxinfer.jl |
| **Output** | analysis/rxinfer | simulation_results.json |

## Tips for AI Assistants

1. **Runtime:** Julia with RxInfer.jl package
2. **Subprocess:** Executes via julia command
3. **Results:** JSON with messages, convergence

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
