# Visualization Plotting

Shared Matplotlib utilities used across all visualization sub-packages.

## Exports

- `plt` — Matplotlib pyplot when available
- `save_plot_safely()` — Safe file writer with error recovery
- `safe_tight_layout()` — Tight layout with fallback for headless rendering
- `MATPLOTLIB_AVAILABLE` — Boolean flag for dependency checking

## Dependencies

- `matplotlib` (optional; plotting calls report unavailable state when absent)
