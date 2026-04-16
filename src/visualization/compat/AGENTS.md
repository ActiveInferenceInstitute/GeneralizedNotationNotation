# Visualization Compatibility Sub-module

## Overview

Compatibility layer that handles optional dependency availability for matplotlib, seaborn, and numpy. Provides safe fallback imports used by all visualization sub-modules.

## Architecture

```
compat/
├── __init__.py        # Package exports (3 lines)
└── viz_compat.py      # Optional dependency detection and fallbacks (40 lines)
```

## Key Exports

- **`MATPLOTLIB_AVAILABLE`** — Boolean flag indicating matplotlib availability.
- **`plt`** — `matplotlib.pyplot` or `None` if unavailable.
- **`sns`** — `seaborn` or `None` if unavailable.
- **`np`** — `numpy` (always available as a core dependency).

## Usage

```python
from visualization.compat.viz_compat import MATPLOTLIB_AVAILABLE, plt, np
if MATPLOTLIB_AVAILABLE:
    fig, ax = plt.subplots()
```

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
