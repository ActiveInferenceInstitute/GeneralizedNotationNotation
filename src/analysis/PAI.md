# Analysis Module - PAI Context

## Quick Reference

**Purpose:** Post-simulation analysis of execution results across all frameworks.

**When to use this module:**
- Analyze simulation outputs from all frameworks
- Compare cross-framework results
- Generate analysis reports and visualizations

## Common Operations

```python
# Run analysis for all frameworks
from analysis.processor import AnalysisProcessor
processor = AnalysisProcessor(input_dir, output_dir)
results = processor.process(verbose=True)

# Post-simulation analysis
from analysis.post_simulation import run_post_simulation_analysis
results = run_post_simulation_analysis(execution_results)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute | simulation_results.json, CSVs |
| **Output** | report | analysis_summary.json, comparisons |

## Framework Analyzers

| Framework | Analyzer | Location |
|-----------|----------|----------|
| PyMDP | `PyMDPAnalyzer` | `analysis/pymdp/` |
| RxInfer.jl | `RxInferAnalyzer` | `analysis/rxinfer/` |
| ActiveInference.jl | `ActiveInferenceJLAnalyzer` | `analysis/activeinference_jl/` |
| JAX | `JAXAnalyzer` | `analysis/jax/` |
| DisCoPy | `DisCoPyAnalyzer` | `analysis/discopy/` |

## Key Files

- `processor.py` - Orchestrates framework-specific analyzers
- `post_simulation.py` - Cross-framework analysis utilities
- `{framework}/analyzer.py` - Framework implementations

## Tips for AI Assistants

1. **Step 16:** Analysis runs after execution, before reporting
2. **Thin Orchestrator:** processor.py delegates to framework analyzers
3. **Output Location:** `output/16_analysis_output/{framework}/`
4. **Comparison:** Compares results across frameworks for same model
5. **Metrics:** beliefs, actions, free energy, convergence

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
