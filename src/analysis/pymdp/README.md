# PyMDP Analysis Module

Framework-specific analysis and visualization for PyMDP simulation results.

## Overview

This submodule provides post-simulation analysis and visualization capabilities for PyMDP Active Inference simulations. It processes execution logs and generates insights, plots, and statistical summaries.

## Module Structure

```
analysis/pymdp/
├── __init__.py      # Module exports
├── analyzer.py      # PyMDPAnalyzer class - log analysis
├── visualizer.py    # PyMDPVisualizer class - plot generation
├── README.md        # This file
└── AGENTS.md        # Agent scaffolding
```

## Key Components

### PyMDPAnalyzer (`analyzer.py`)

Analyzes PyMDP simulation logs and extracts metrics:

```python
from analysis.pymdp.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/pymdp"),
    verbose=True
)
```

**Capabilities:**
- Parse simulation trace files
- Extract belief dynamics
- Analyze action distributions
- Calculate free energy metrics
- Generate statistical summaries

### PyMDPVisualizer (`visualizer.py`)

Generates visualizations from simulation data:

```python
from analysis.pymdp.visualizer import PyMDPVisualizer

viz = PyMDPVisualizer(output_dir=Path("output/visualizations"))
viz.plot_beliefs(trace_data)
viz.plot_actions(trace_data)
viz.plot_free_energy(trace_data)
```

**Plot Types:**
- Belief state evolution
- Action probability distributions
- Free energy over time
- Observation sequences
- State-action heatmaps

## Data Flow

```
PyMDP Execution Results (Step 12)
    ↓
PyMDPAnalyzer.analyze()
    ↓
├── Trace parsing
├── Metric extraction
└── Statistical analysis
    ↓
PyMDPVisualizer.visualize()
    ↓
├── Belief plots
├── Action plots
└── Free energy plots
    ↓
Analysis Output (Step 16)
```

## Input/Output

**Input:** `output/12_execute_output/*/pymdp_gen/`
- `simulation_results.json`
- Execution logs
- Trace files

**Output:** `output/16_analysis_output/pymdp/`
- `analysis_summary.json`
- Visualization PNG files
- Statistical reports

---

**Framework:** PyMDP (Python)
**Version:** 1.1.3
**Status:** Production Ready
