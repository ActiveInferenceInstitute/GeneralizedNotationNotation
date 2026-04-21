---
name: gnn-advanced-viz
description: GNN advanced visualization and interactive plots. Use when creating D2 diagrams, dashboards, interactive network visualizations, timeline charts, heatmaps, or data extraction for custom visualizations.
---

# GNN Advanced Visualization (Step 9)

## Purpose

Generates advanced visualizations including D2 diagrams, interactive dashboards, network and timeline visualizations, heatmaps, and data extraction for GNN models.

## Key Commands

```bash
# Run advanced visualization
python src/9_advanced_viz.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 9 --verbose
```

## API

```python
from advanced_visualization import (
    AdvancedVisualizer,
    create_visualization_from_data, create_network_visualization,
    create_timeline_visualization, create_heatmap_visualization,
    DashboardGenerator, generate_dashboard,
    VisualizationDataExtractor, extract_visualization_data,
    process_advanced_viz,
)

# Create visualizations
viz = AdvancedVisualizer()
create_network_visualization(data, output_path="net.html")
create_heatmap_visualization(data, output_path="heat.html")

# Generate dashboards
gen = DashboardGenerator()
generate_dashboard(model_data, output_dir="output/")

# Extract visualization data
extractor = VisualizationDataExtractor()
viz_data = extract_visualization_data(parsed_model)

# D2 diagram generation (if available)
from advanced_visualization import D2Visualizer, D2_AVAILABLE
if D2_AVAILABLE:
    d2 = D2Visualizer()
```

## Key Exports

- `AdvancedVisualizer` — main advanced visualization class
- `DashboardGenerator` / `generate_dashboard` — interactive HTML dashboards
- `VisualizationDataExtractor` / `extract_visualization_data` — data preparation
- `D2Visualizer` / `D2DiagramSpec` / `D2GenerationResult` — D2 diagram support
- `create_network_visualization`, `create_timeline_visualization`, `create_heatmap_visualization`

## Dependencies

```bash
# Core interactive visualization
uv sync --extra visualization

# For D2 diagrams (requires d2 CLI)
# Install d2: https://d2lang.com/tour/install
```

## Output

- Interactive HTML files in `output/9_advanced_viz_output/`
- D2 diagrams (when D2 is available)
- Dashboard pages


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_advanced_visualization`
- `check_visualization_capabilities`
- `list_d2_visualization_types`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
