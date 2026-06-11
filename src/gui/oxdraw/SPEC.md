# OxDraw — Technical Specification

**Version**: 1.6.0

## Purpose

Oxford-style drawing tool for Active Inference graphical models.

## Features

- Forney-style factor graph rendering
- Plate notation for repeated structures
- Factor node customization
- Export to SVG and PNG

## Architecture

```
oxdraw/
├── __init__.py      # Package exports (161 lines)
├── renderer.py      # Core rendering engine
├── elements.py      # Graph element definitions
├── layout.py        # Graph layout algorithms
├── export.py        # SVG/PNG export
└── styles.py        # Visual styling configuration
```

## Technology

- Pure Python rendering
- Optional `cairo` for high-quality output
