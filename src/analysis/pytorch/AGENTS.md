# PyTorch Analysis Sub-module

## Overview

Framework-specific analysis module for PyTorch simulation outputs. Reads `simulation_results.json` produced by the PyTorch runner and generates belief trajectory, action distribution, and EFE analysis plots.

## Architecture

```
pytorch/
├── __init__.py      # Package exports
└── analyzer.py      # PyTorch result analysis and visualization (186 lines)
```

## Key Functions

- **`analyze_pytorch_results(results_dir, output_dir)`** — Main entry point; reads PyTorch JSON results and produces matplotlib visualizations.
- **Belief trajectory plotting** — Tracks posterior belief evolution over simulation timesteps.
- **Action distribution analysis** — Visualizes policy distributions across actions.
- **EFE decomposition** — Breaks down Expected Free Energy into epistemic and instrumental components.

## Dependencies

- `numpy`, `matplotlib` (required)
- `torch` (runtime, for result interpretation)

## Parent Module

See [analysis/AGENTS.md](../AGENTS.md) for the overall analysis architecture.

**Version**: 1.6.0
