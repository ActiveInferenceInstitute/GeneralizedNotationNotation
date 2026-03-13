# Step 9: Advanced Visualization — 3D, Interactive, and POMDP Viz

## Overview

Generates advanced visualizations including 3D plots, interactive dashboards, D2 diagrams, statistical analysis plots, POMDP-specific visualizations, and network analysis graphics.

## Usage

```bash
# All visualization types
python src/9_advanced_viz.py --target-dir input/gnn_files --output-dir output --verbose

# Specific type
python src/9_advanced_viz.py --viz-type pomdp --verbose
python src/9_advanced_viz.py --viz-type dashboard --interactive --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/9_advanced_viz.py` (53 lines) |
| Module | `src/advanced_visualization/` |
| Processor | `src/advanced_visualization/processor.py` |
| Module function | `process_advanced_viz_standardized_impl()` |

Uses direct import — no try/except handlers.

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--viz-type` | `str` | `all` | Visualization type: `all`, `3d`, `interactive`, `dashboard`, `d2`, `diagrams`, `pipeline`, `statistical`, `pomdp`, `network` |
| `--interactive` | `bool` | `True` | Generate interactive visualizations |
| `--export-formats` | `str[]` | `html json` | Export formats for outputs |

## Output

- **Directory**: `output/9_advanced_viz_output/`
- HTML interactive dashboards, 3D plots, D2 diagrams, statistical plots, POMDP visualizations

## Source

- **Script**: [src/9_advanced_viz.py](#placeholder)
