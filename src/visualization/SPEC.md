# Visualization Module Specification

## Overview
Core visualization generation for GNN models.

## Components

### Core
- `processor.py` - Main visualization processor (1337 lines)
- `matrix_visualizer.py` - Matrix visualization (1415 lines)

### Parsing
- `parser.py` - GNN visualization parser

## Visualization Types
- Matrix heatmaps
- Connection graphs
- State space visualizations

## Key Exports
```python
from visualization import process_visualization_main, MatrixVisualizer
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
