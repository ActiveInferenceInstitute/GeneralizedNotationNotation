---
name: gnn-visualization
description: GNN graph and matrix visualization generation. Use when creating network graph plots, matrix heatmaps, state space diagrams, or other visual representations of GNN models.
---

# GNN Visualization (Step 8)

## Purpose

Generates static visualizations of GNN models including network graphs, matrix heatmaps, state space diagrams, and connection maps using matplotlib and networkx.

## Key Commands

```bash
# Run visualization
python src/8_visualization.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 8 --verbose
```

## API

```python
from pathlib import Path
from visualization import (
    GNNVisualizer, MatrixVisualizer, GraphVisualizer,
    generate_graph_visualization, generate_matrix_visualization,
    generate_visualizations, process_visualization,
)

# Use the GNNVisualizer class
viz = GNNVisualizer(output_dir="output/")
viz.generate_graph_visualization(graph_data)
viz.generate_matrix_visualization(matrix_data)

# Standalone functions
generate_graph_visualization(graph_data, output_dir="output/")
generate_matrix_visualization(matrix_data, output_dir="output/")

# Run full visualization step (used by pipeline)
process_visualization(Path("input/gnn_files"), Path("output/8_visualization_output"), verbose=True)
```

## Key Exports

- `GNNVisualizer` / `GraphVisualizer` — main visualization classes
- `MatrixVisualizer` — matrix-specific visualization
- `generate_graph_visualization` — network graph plots
- `generate_matrix_visualization` — matrix heatmaps
- `process_visualization` — pipeline processing (JSON-first when step-3 `*_parsed.json` exists)

## Safe-to-Fail Pattern

Visualization uses a four-tier recovery system:

1. Full visualizer with all dependencies
2. Matrix-only visualizer
3. Basic plots
4. HTML recovery report

## Dependencies

```bash
# matplotlib + networkx (included in core)
uv sync

# For additional viz backends
uv sync --extra visualization
```

## Output

- PNG/SVG images in `output/8_visualization_output/`
- HTML recovery reports when matplotlib unavailable


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_visualization_module_info`
- `get_visualization_options`
- `list_visualization_artifacts`
- `process_visualization`

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
