# Visualization Compatibility — Technical Specification

**Version**: 1.6.0

## Import Strategy

```python
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
```

## Contract

- `MATPLOTLIB_AVAILABLE` is always a `bool`
- `plt` is either `matplotlib.pyplot` or `None`
- `np` is always `numpy` (core dependency)
