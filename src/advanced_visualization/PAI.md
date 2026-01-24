# Advanced Visualization Module - PAI Context

## Quick Reference

**Purpose:** Advanced interactive visualizations including D2 diagrams and dashboards.

**When to use this module:**
- Generate D2 architecture diagrams
- Create interactive HTML dashboards
- Build complex multi-panel visualizations

## Common Operations

```python
# Generate advanced visualizations
from advanced_visualization.processor import AdvancedVisualizationProcessor
processor = AdvancedVisualizationProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | All prior steps | Results, metrics |
| **Output** | report | D2, HTML dashboards |

## Key Files

- `processor.py` - Main processor class
- `d2_generator.py` - D2 diagram generation
- `dashboard.py` - Interactive dashboard creation
- `__init__.py` - Public API exports

## Visualization Types

| Type | Format | Description |
|------|--------|-------------|
| D2 Diagrams | .d2 | Architecture diagrams |
| Dashboard | HTML | Interactive exploration |
| Flow Charts | SVG | Pipeline flow visualization |

## Tips for AI Assistants

1. **Step 9:** Advanced visualization is Step 9
2. **D2:** Uses D2 language for diagrams
3. **Output Location:** `output/9_advanced_viz_output/`
4. **Interactive:** Dashboards support filtering and drill-down

---

**Version:** 1.1.3 | **Step:** 9 (Advanced Visualization)
