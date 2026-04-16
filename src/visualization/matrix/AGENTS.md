# Visualization Matrix Sub-module

## Overview

Matrix visualization module for GNN model parameters. Generates heatmaps, statistics, and analysis plots for transition matrices, observation matrices, and 3D POMDP tensors.

## Architecture

```
matrix/
├── __init__.py       # Package exports (9 lines)
├── visualizer.py     # Matrix heatmap and statistics engine (1338 lines)
├── extract.py        # Matrix extraction from parsed models (60 lines)
└── compat.py         # Backward compatibility helpers (40 lines)
```

## Key Functions

- **`visualize_matrices(model, output_dir)`** — Main entry point; extracts and visualizes all matrices from a parsed GNN model.
- **Heatmap generation** — Produces annotated heatmaps with value labels, colorbars, and statistical summaries.
- **3D tensor handling** — Specialized support for POMDP transition tensors with per-slice visualization.
- **CSV export** — Exports matrix data alongside visualizations for downstream analysis.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
