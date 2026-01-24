# ActiveInference.jl Analysis Module

Framework-specific analysis for ActiveInference.jl simulation results.

## Overview

This submodule provides analysis and visualization capabilities for ActiveInference.jl (Julia) simulations. It includes both Python analyzers for pipeline integration and Julia analysis suites for advanced statistical analysis.

## Module Structure

```
analysis/activeinference_jl/
├── __init__.py                      # Module exports
├── analyzer.py                      # Python analyzer for pipeline integration
├── analysis_suite.jl                # Julia analysis suite
├── advanced_pomdp_analysis.jl       # Advanced POMDP analysis
├── statistical_analysis.jl          # Statistical methods
├── uncertainty_quantification.jl    # Uncertainty analysis
├── multi_scale_temporal_analysis.jl # Temporal dynamics
├── meta_cognitive_analysis.jl       # Meta-cognitive analysis
├── visualization_suite.jl           # Julia visualization
└── visualization_utils.jl           # Visualization helpers
```

## Key Components

### Python Analyzer (`analyzer.py`)

Pipeline-integrated analyzer:

```python
from analysis.activeinference_jl.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/activeinference_jl"),
    verbose=True
)
```

### Julia Analysis Suites

Advanced analysis in native Julia:

- **analysis_suite.jl**: Core analysis functions
- **advanced_pomdp_analysis.jl**: POMDP-specific analysis
- **statistical_analysis.jl**: Statistical methods
- **uncertainty_quantification.jl**: Bayesian uncertainty
- **multi_scale_temporal_analysis.jl**: Multi-scale dynamics
- **meta_cognitive_analysis.jl**: Higher-order inference analysis

## Capabilities

- Free energy convergence analysis
- Belief precision tracking
- Action selection analysis
- Multi-scale temporal dynamics
- Uncertainty quantification
- Meta-cognitive metrics

## Data Flow

```
ActiveInference.jl Execution (Step 12)
    ↓
Python Analyzer (pipeline integration)
    ↓
├── Parse simulation outputs
├── Extract metrics
└── Generate summaries
    ↓
Julia Analysis (optional deep analysis)
    ↓
├── Advanced statistics
├── Uncertainty quantification
└── Multi-scale analysis
    ↓
Analysis Output (Step 16)
```

---

**Framework:** ActiveInference.jl (Julia)
**Version:** 1.1.3
**Status:** Production Ready
