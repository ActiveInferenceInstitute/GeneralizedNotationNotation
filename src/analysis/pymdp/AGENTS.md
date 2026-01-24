# PyMDP Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for PyMDP simulation results. Part of the Analysis module (Step 16).

## Module Structure

```
analysis/pymdp/
├── __init__.py      # Public API
├── analyzer.py      # PyMDPAnalyzer class
├── visualizer.py    # PyMDPVisualizer class
├── README.md        # Human documentation
└── AGENTS.md        # This file
```

## Key Classes

### PyMDPAnalyzer

**Location:** `analyzer.py`

**Purpose:** Parse and analyze PyMDP simulation traces

**Key Methods:**
- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point
- `_parse_trace_file(filepath)` - Load simulation trace
- `_analyze_beliefs(trace)` - Extract belief dynamics
- `_analyze_actions(trace)` - Action distribution analysis
- `_calculate_metrics(trace)` - Compute statistical metrics

### PyMDPVisualizer

**Location:** `visualizer.py`

**Purpose:** Generate plots from simulation data

**Key Methods:**
- `plot_beliefs(trace_data)` - Belief evolution plots
- `plot_actions(trace_data)` - Action distribution plots
- `plot_free_energy(trace_data)` - Free energy convergence
- `save_all_visualizations(trace_data, output_dir)` - Batch save

## Integration Points

**Upstream:** Execute module (Step 12) produces simulation results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

- numpy: Numerical operations
- matplotlib: Visualization (optional)
- pandas: Data manipulation (optional)

---

**Version:** 1.1.3
**Last Updated:** 2026-01-23
