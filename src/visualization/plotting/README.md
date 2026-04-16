# Visualization Plotting

Shared Matplotlib utilities used across all visualization sub-packages.

## Exports

- `plt` — Matplotlib pyplot (or stub if unavailable)
- `save_plot_safely()` — Safe file writer with error recovery
- `safe_tight_layout()` — Tight layout with fallback for headless rendering
- `MATPLOTLIB_AVAILABLE` — Boolean flag for dependency checking

## Dependencies

- `matplotlib` (optional; provides stubs when absent)
