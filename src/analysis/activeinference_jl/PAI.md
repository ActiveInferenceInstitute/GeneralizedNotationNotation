# ActiveInference.jl Analysis - PAI Context

## Quick Reference

**Purpose:** Analyze ActiveInference.jl (Julia) simulation results for Active Inference models.

**When to use this module:**
- Post-simulation analysis of ActiveInference.jl outputs
- Extracting belief trajectories from Julia CSV outputs
- Generating analysis reports for Step 16

## Common Operations

```python
# Analyze ActiveInference.jl execution results
from analysis.activeinference_jl.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/activeinference_jl"),
    verbose=True
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute/activeinference_jl | simulation_results.csv |
| **Output** | report | analysis_summary.json |

## Key Files

- `analyzer.py` - Main `ActiveInferenceJLAnalyzer` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Data Location:** Results in `output/12_execute_output/*/activeinference_jl/`
2. **Output Format:** CSV with columns: step, observation, action, belief_state_*
3. **Julia Execution:** Requires Julia runtime with ActiveInference.jl package
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
