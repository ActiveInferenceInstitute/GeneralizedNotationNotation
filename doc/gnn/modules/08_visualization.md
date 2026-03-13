# Step 8: Visualization — Matrix and Network Graphs

## Overview

Generates visualizations from parsed GNN specifications including matrix heatmaps, network graphs, and combined analysis plots.

## Usage

```bash
python src/8_visualization.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/8_visualization.py` (56 lines) |
| Module | `src/visualization/` |
| Processor | `src/visualization/processor.py` |
| Module function | `process_visualization_main()` |

Uses direct import — no try/except handlers.

## Output

- **Directory**: `output/8_visualization_output/`
- Matrix heatmaps (`.png`), network graphs, and combined analysis plots

## Source

- **Script**: [src/8_visualization.py](#placeholder)
