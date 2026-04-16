# PyMDP Renderer — Technical Specification

**Version**: 1.6.0

## Purpose

Generates Python code for PyMDP POMDP simulation.

## Code Generation

- Maps GNN model structure to PyMDP agent constructor
- Generates transition matrices (A, B), preference vectors (C), prior (D)
- Produces standalone simulation scripts

## Output

- Python scripts using `pymdp` API
- JSON parameter files

## Architecture

```
pymdp/
├── __init__.py             # Package exports
├── pymdp_converter.py      # GNN-to-PyMDP conversion (1512 lines)
├── pymdp_templates.py      # Code templates (527 lines)
└── ...
```

## API Compatibility

- Supports PyMDP 0.x and 1.0.0+ API variants
- Auto-detects target version via `execute/pymdp/package_detector.py`
