# Visualization Analysis Sub-module

## Overview

Combined analysis visualization module that aggregates results from multiple pipeline steps into unified analytical views.

## Architecture

```
analysis/
├── __init__.py              # Package exports (6 lines)
└── combined_analysis.py     # Multi-source analysis visualization (508 lines)
```

## Key Functions

- **`generate_combined_analysis(data_sources, output_dir)`** — Produces combined visualizations from GNN parse results, execution outputs, and validation reports.
- **Cross-step aggregation** — Merges data from Steps 7 (export), 8 (visualization), 12 (execution), and 16 (analysis).
- **Summary dashboards** — Generates HTML dashboards showing model health across all pipeline phases.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
