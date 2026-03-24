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

1. **Data Location (Step 12):** After `process_execute`, artifacts are under  
   `output/12_execute_output/<gnn_file_stem>/pymdp/`:
   - `simulation_data/` — collected `simulation_results.json` (and prefixed copies if needed)
   - `execution_logs/` — per-script stdout/stderr and `*_results.json`
   Raw simulator output may also exist under the render tree at  
   `<model>/pymdp/output/pymdp_simulations/<model_name>/` before collection copies it into `simulation_data/`.
2. **Output Format (Step 16):** Framework plots and tables under `output/16_analysis_output/pymdp/<model_slug>/`.
3. **Key Metrics:** beliefs, actions, observations, expected free energy (from `simulation_trace` / `metrics`).
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly.

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
