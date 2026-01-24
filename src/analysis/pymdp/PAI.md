# PyMDP Analysis - PAI Context

## Quick Reference

**Purpose:** Analyze PyMDP (Python) simulation results for Active Inference models.

**When to use this module:**
- Post-simulation analysis of PyMDP outputs
- Extracting belief trajectories and action sequences
- Generating analysis reports for Step 16

## Common Operations

```python
# Analyze PyMDP execution results
from analysis.pymdp.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/pymdp"),
    verbose=True
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute/pymdp | simulation_results.json |
| **Output** | report | analysis_summary.json |

## Key Files

- `analyzer.py` - Main `PyMDPAnalyzer` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Data Location:** PyMDP results are in `output/12_execute_output/*/pymdp_gen/`
2. **Output Format:** Analysis goes to `output/16_analysis_output/pymdp/`
3. **Key Metrics:** beliefs, actions, observations, free energy values
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
