# Visualization Compatibility

Safe dependency detection for optional visualization libraries (matplotlib, seaborn).

## Files

- `viz_compat.py` — Exports `MATPLOTLIB_AVAILABLE`, `plt`, `sns`, `np`; `sns` resolves lazily via `get_sns()` on first access, so importing this module never imports seaborn.

## Usage

```python
from gnn.visualization.compat.viz_compat import MATPLOTLIB_AVAILABLE, plt
```

## See Also

- [Parent: visualization/README.md](../README.md)
