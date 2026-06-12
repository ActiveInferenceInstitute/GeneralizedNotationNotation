# Visualization Analysis Sub-module

## Overview

Combined analysis visualization module producing multi-panel summary plots, standalone panels, generative model diagrams, and cross-file comparison charts.

## Architecture

```
analysis/
├── __init__.py              # Package exports
└── combined_analysis.py     # Multi-source analysis visualization
```

## Key Functions

- **`generate_combined_analysis(parsed_data, output_dir, model_name) -> List[str]`** — 2×2 panel grid: variable type distribution, connection count histogram, matrix size distribution, section content length.
- **`generate_combined_visualizations(gnn_files, results_dir, verbose) -> List[str]`** — Cross-file aggregate analysis: overall variable distribution, file comparison, matrix sizes, top connection types.

### Internal Helpers

- `_generate_standalone_panels(parsed_data, output_dir, model_name)` — Individual full-page charts for matrix size, section length, and variable type.
- `_generate_generative_model_diagram(parsed_data, output_dir, model_name)` — POMDP circuit diagram (D, s, A, o, B, C, E, π, G, u) with annotated edges and legend.

## Dependencies

- Uses `visualization.core.parsed_model.load_visualization_model` for JSON-first model loading.
- Uses `visualization.plotting.utils.save_plot_safely` / `safe_tight_layout` for robust file saving.
- Uses `visualization.compat.viz_compat.viz_var_type` for variable type extraction (canonicalized in compat).

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
