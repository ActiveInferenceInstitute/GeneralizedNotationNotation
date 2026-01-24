# Visualization Module - PAI Context

## Quick Reference

**Purpose:** Generate visualizations of model structure and simulation results.

**When to use this module:**
- Create network diagrams of model structure
- Visualize belief trajectories
- Generate state-space plots

## Common Operations

```python
# Generate visualizations
from visualization.processor import VisualizationProcessor
processor = VisualizationProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn, execute | Model structure, results |
| **Output** | report | PNG, SVG visualizations |

## Key Files

- `processor.py` - Main `VisualizationProcessor` class
- `__init__.py` - Public API exports

## Visualization Types

| Type | Description |
|------|-------------|
| Network | Model state-space graph |
| Trajectory | Belief evolution over time |
| Matrix | A, B, C, D matrix heatmaps |

## Tips for AI Assistants

1. **Step 8:** Visualization is Step 8 of the pipeline
2. **Dependencies:** Uses matplotlib, networkx
3. **Output Location:** `output/8_visualization_output/`
4. **Formats:** PNG and SVG supported

---

**Version:** 1.1.3 | **Step:** 8 (Visualization)
