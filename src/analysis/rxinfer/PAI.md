# RxInfer Analysis - PAI Context

## Quick Reference

**Purpose:** Analyze RxInfer.jl (Julia) message-passing simulation results for Bayesian Active Inference models.

**When to use this module:**
- Post-simulation analysis of RxInfer.jl outputs
- Extracting message convergence and belief propagation metrics
- Generating analysis reports for Step 16

## Common Operations

```python
# Analyze RxInfer execution results
from analysis.rxinfer.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/rxinfer"),
    verbose=True
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute/rxinfer | simulation_results.json |
| **Output** | report | analysis_summary.json |

## Key Files

- `analyzer.py` - Main `RxInferAnalyzer` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Data Location:** RxInfer results in `output/12_execute_output/*/rxinfer/`
2. **Key Metrics:** message counts, convergence iterations, belief updates
3. **Factor Graphs:** RxInfer uses reactive message passing on factor graphs
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
