# Visualization Plotting Sub-module

## Overview

Shared plotting utilities used across all visualization sub-modules. Provides consistent styling, color palettes, and common matplotlib configuration.

## Architecture

```
plotting/
├── __init__.py     # Plotting exports and configuration (19 lines)
└── utils.py        # Shared plotting utilities (73 lines)
```

## Key Functions

- **`setup_plot_style()`** — Configures matplotlib rcParams for consistent GNN pipeline styling.
- **`get_color_palette(n)`** — Returns a harmonious color palette of `n` colors for multi-series plots.
- **`save_figure(fig, output_path, formats)`** — Saves figures in multiple formats (PNG, SVG, PDF) with consistent DPI settings.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
