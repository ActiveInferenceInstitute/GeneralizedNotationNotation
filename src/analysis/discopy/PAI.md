# DisCoPy Analysis - PAI Context

## Quick Reference

**Purpose:** Analyze DisCoPy categorical diagram results for compositional Active Inference models.

**When to use this module:**
- Post-simulation analysis of DisCoPy diagram outputs
- Extracting circuit structure and composition metrics
- Generating analysis reports for Step 16

## Common Operations

```python
# Analyze DisCoPy execution results
from analysis.discopy.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/discopy"),
    verbose=True
)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute/discopy | circuit_info.json, diagrams |
| **Output** | report | analysis_summary.json |

## Key Files

- `analyzer.py` - Main `DisCoPyAnalyzer` class
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Data Location:** DisCoPy results in `output/12_execute_output/*/discopy/`
2. **Key Metrics:** diagram topology, box/wire counts, composition depth
3. **Category Theory:** Based on categorical quantum mechanics formalism
4. **Thin Orchestrator:** Called by `analysis/processor.py`, not directly

---

**Version:** 1.1.3 | **Step:** 16 (Analysis)
