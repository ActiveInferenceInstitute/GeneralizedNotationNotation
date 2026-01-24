# ActiveInference.jl Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for ActiveInference.jl simulation results. Part of the Analysis module (Step 16).

## Module Structure

```
analysis/activeinference_jl/
├── __init__.py                      # Public API
├── analyzer.py                      # Python pipeline analyzer
├── analysis_suite.jl                # Julia analysis core
├── advanced_pomdp_analysis.jl       # POMDP analysis
├── statistical_analysis.jl          # Statistics
├── uncertainty_quantification.jl    # Uncertainty
├── multi_scale_temporal_analysis.jl # Temporal
├── meta_cognitive_analysis.jl       # Meta-cognitive
├── visualization_suite.jl           # Visualization
└── visualization_utils.jl           # Utils
```

## Key Functions

### Python (`analyzer.py`)

- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point

### Julia Analysis Suites

**analysis_suite.jl:**
- Core analysis functions for ActiveInference.jl results

**advanced_pomdp_analysis.jl:**
- POMDP-specific metrics and analysis

**statistical_analysis.jl:**
- Statistical tests and distributions

**uncertainty_quantification.jl:**
- Bayesian uncertainty metrics

**multi_scale_temporal_analysis.jl:**
- Multi-scale temporal dynamics

**meta_cognitive_analysis.jl:**
- Higher-order inference analysis

## Integration Points

**Upstream:** Execute module (Step 12) produces Julia simulation results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

**Python:**
- pathlib, json, logging

**Julia:**
- ActiveInference.jl
- Distributions.jl
- Plots.jl (optional)

---

**Version:** 1.1.3
**Last Updated:** 2026-01-23
