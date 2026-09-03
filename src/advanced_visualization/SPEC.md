# Advanced Visualization Module Specification

## Overview
Advanced visualization generation including 3D, interactive dashboards, and D2 diagrams.

## Components

### Core
- `processor.py` - Main processor (573 lines); `process_advanced_viz` entry point
- `visualizer.py` - Visualization generation (`AdvancedVisualizer`)
- `dashboard.py` - Dashboard generation (`DashboardGenerator`)

### D2 Integration
- `d2_visualizer.py` - D2 diagram generation (`D2Visualizer`)
- `D2_README.md` - D2 documentation

### Support
- `data_extractor.py` - Data extraction utilities (`VisualizationDataExtractor`)
- `html_generator.py` - HTML report generation

## Key Exports
```python
from advanced_visualization import process_advanced_viz
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
