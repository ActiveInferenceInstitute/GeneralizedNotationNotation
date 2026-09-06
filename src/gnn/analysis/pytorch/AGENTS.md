# PyTorch Analysis Sub-module

## Overview

Framework-specific analysis module for PyTorch simulation outputs. Reads `simulation_results.json` produced by the PyTorch runner and generates belief trajectory, action distribution, and EFE analysis plots.

## Architecture

```
pytorch/
├── __init__.py      # Package exports
└── analyzer.py      # PyTorch result analysis and visualization (202 lines)
```

## Key Functions

- **`generate_analysis_from_logs(results_dir, output_dir=None, verbose=False)`** — Main entry point; reads PyTorch JSON results and produces matplotlib visualizations.
- **Belief trajectory plotting** — Tracks posterior belief evolution over simulation timesteps.
- **Action distribution analysis** — Visualizes policy distributions across actions.

## Dependencies

- `numpy`, `matplotlib` (required)
- `torch` (runtime, for result interpretation)

## Parent Module

See [analysis/AGENTS.md](../AGENTS.md) for the overall analysis architecture.

**Version**: 3.2.0
