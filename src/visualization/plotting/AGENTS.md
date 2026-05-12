# Visualization Plotting Sub-module

## Overview

Shared plotting utilities used across all visualization sub-modules. Provides safe matplotlib save routines with DPI fallbacks and warning-suppressed tight layout.

## Architecture

```
plotting/
├── __init__.py     # Package exports and configuration
└── utils.py        # Shared plotting utilities
```

## Key Functions

- **`save_plot_safely(plot_path, dpi=300, **savefig_kwargs) -> bool`** — Save current figure with automatic DPI fallback chain (user → rcParams → no-DPI). Returns `True` on success.
- **`safe_tight_layout() -> None`** — Apply `plt.tight_layout()` with `UserWarning` suppression. Silently skips on `ValueError` / `RuntimeError`.

### Backward Compatibility

- `_save_plot_safely` — Alias for `save_plot_safely` (used by legacy processor shims).
- `_safe_tight_layout` — Alias for `safe_tight_layout`.

## Usage

```python
from visualization.plotting.utils import save_plot_safely, safe_tight_layout

safe_tight_layout()
save_plot_safely(output_path, dpi=300, bbox_inches="tight")
```

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
