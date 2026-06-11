# Visualization Compatibility Sub-module

## Overview

Compatibility layer that handles optional dependency availability for matplotlib, seaborn, and numpy. Provides safe fallback imports and shared helpers used by all visualization sub-modules.

## Architecture

```
compat/
├── __init__.py        # Package exports
└── viz_compat.py      # Optional dependency detection, fallbacks, and shared helpers
```

## Key Exports

- **`MATPLOTLIB_AVAILABLE`** — Boolean flag indicating matplotlib availability.
- **`plt`** — `matplotlib.pyplot` or `None` if unavailable.
- **`sns`** — `seaborn` or `None` if unavailable.
- **`np`** — `numpy` (always available as a core dependency).
- **`viz_var_type(var_info: dict) -> str`** — Extract variable type from a parsed variable dict. Checks `var_type`, `type`, and `node_type` keys in order, returning `"unknown"` when none are present. Canonical source — imported by `analysis/combined_analysis.py` and `graph/network_visualizations.py`.

## Usage

```python
from visualization.compat.viz_compat import MATPLOTLIB_AVAILABLE, plt, np, viz_var_type

if MATPLOTLIB_AVAILABLE:
    fig, ax = plt.subplots()

var_type = viz_var_type({"var_type": "hidden_state"})  # -> "hidden_state"
```

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
