# Visualization Module Specification

## Overview

Step-8 visualization: graphs, matrices, combined analysis. Prefers step-3 `{model}_parsed.json`; falls back to markdown parsing.

## Layout

| Package / file | Responsibility |
|----------------|----------------|
| `core/process.py` | Orchestration, per-file pipeline, sampling |
| `core/parsed_model.py` | `load_visualization_model`, step-3 path resolution |
| `parse/markdown.py` | `parse_gnn_content` fallback |
| `parse/gnn_file_parser.py` | `GNNParser` (file / CSV-oriented API) |
| `plotting/utils.py` | Matplotlib Agg, save/tight_layout helpers |
| `graph/network_visualizations.py` | Directed vs undirected edges, ontology labels |
| `graph/bipartite.py` | Variable–parameter bipartite sketch |
| `matrix/visualizer.py` | `MatrixVisualizer` facade, module `generate_matrix_visualizations` |
| `matrix/extract.py` | `convert_to_matrix`, parameter extraction |
| `matrix/compat.py` | `parse_matrix_data` (markdown matrices), batch `generate_matrix_visualizations` |
| `analysis/combined_analysis.py` | Combined and cross-file plots |
| `ontology/visualizer.py` | Ontology table / annotations |
| `compat/viz_compat.py` | Shared `plt` / `np` / `sns` for step 8 and analysis |
| Root shims | `processor.py`, `matrix_visualizer.py`, `parser.py`, `network_visualizations.py`, `combined_analysis.py`, `ontology_visualizer.py`, `_viz_compat.py` |

## Visualization types

- Matrix heatmaps and 3D POMDP transition views
- Connection graphs (respecting `connection_type`)
- Combined analysis and generative-model schematic
- Optional Plotly HTML network

## Key exports

```python
from visualization import process_visualization, MatrixVisualizer
from visualization.processor import parse_gnn_content, _save_plot_safely
from visualization.parse import GNNParser, parse_gnn_content
```

---

## Documentation

- **[README](README.md)**: Module overview
- **[AGENTS](AGENTS.md)**: Agent workflows
- **[SPEC](SPEC.md)**: This file
- **[SKILL](SKILL.md)**: Capability API
