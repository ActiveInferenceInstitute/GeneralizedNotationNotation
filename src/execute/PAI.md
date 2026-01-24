# Execute Module - PAI Context

## Quick Reference

**Purpose:** Execute rendered framework code and capture simulation results.

**When to use this module:**
- Run PyMDP Python simulations
- Run RxInfer.jl Julia simulations
- Run ActiveInference.jl Julia simulations
- Run JAX Python simulations
- Run DisCoPy Python simulations

## Common Operations

```python
# Execute all frameworks
from execute.processor import ExecuteProcessor
processor = ExecuteProcessor(input_dir, output_dir)
results = processor.process(verbose=True)

# Execute specific framework
from execute.pymdp.executor import PyMDPExecutor
executor = PyMDPExecutor()
results = executor.execute(code_path, output_dir)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render | Framework code files |
| **Output** | analysis | simulation_results.json, CSVs |

## Framework Executors

| Framework | Executor | Runtime |
|-----------|----------|---------|
| PyMDP | `pymdp/executor.py` | Python |
| RxInfer.jl | `rxinfer/rxinfer_runner.jl` | Julia |
| ActiveInference.jl | `activeinference_jl/*.jl` | Julia |
| JAX | `jax/jax_runner.py` | Python + JAX |
| DisCoPy | `discopy/discopy_runner.py` | Python |

## Key Files

- `processor.py` - Orchestrates all executors
- `{framework}/executor.py` - Framework implementations
- `{framework}/*_runner.{py,jl}` - Runtime scripts

## Tips for AI Assistants

1. **Step 12:** Execute runs after render, before analysis
2. **Subprocess:** Uses `subprocess.run()` for isolation
3. **Output Location:** `output/12_execute_output/{model}/{framework}/`
4. **Result Capture:** Extracts beliefs, actions, observations from outputs
5. **Julia Requirements:** RxInfer and ActiveInference.jl need Julia installed

---

**Version:** 1.1.3 | **Step:** 12 (Execute)
