# JAX Analysis - PAI Context

## Quick Reference

**Purpose:** Analyze JAX simulation results for differentiable Active Inference models.

**When to use this module:**
- Post-simulation analysis of JAX outputs
- Extracting gradient information and convergence metrics
- Generating analysis reports for Step 16

## Common Operations

```python
# Analyze JAX execution results
from analysis.jax.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/jax"),
    verbose=True
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute/jax | jax_outputs/*.json |
| **Output** | report | analysis_summary.json |

## Key Files

- `analyzer.py` - Main `JAXAnalyzer` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Data Location:** JAX results in `output/12_execute_output/*/jax/`
2. **Key Metrics:** gradients, convergence, loss trajectories
3. **Differentiable:** JAX provides automatic differentiation for optimization
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
