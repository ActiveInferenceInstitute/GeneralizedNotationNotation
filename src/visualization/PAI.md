# Visualization Module - PAI Context

## Quick Reference

**Purpose:** Generate visualizations of GNN model structure (graphs, matrices, combined panels).

**Imports:** From the repo root, use `uv run python` (or `PYTHONPATH=src`) so top-level packages resolve; `from visualization import process_visualization`. Full API: [AGENTS.md](AGENTS.md).

**When to use this module:**

- Network diagrams with GNN connection types (directed / undirected)
- Matrix and tensor heatmaps from step-3 `*_parsed.json` when present
- Combined analysis figures and optional variable–parameter bipartite view

## Common Operations

```python
from pathlib import Path
from visualization import process_visualization

ok = process_visualization(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/8_visualization_output"),
    verbose=True,
)
```

Data loading is **JSON-first**: `{model}/{model}_parsed.json` from step 3 when available; otherwise markdown is parsed via `visualization.parse.markdown.parse_gnn_content`.

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | `gnn` (step 3), GNN `.md` files | Parsed JSON or raw markdown |
| **Output** | `report`, `website` | PNG, JSON stats, HTML (plotly optional) |

## Key layout

| Path | Role |
|------|------|
| `core/process.py` | `process_visualization`, `process_single_gnn_file` |
| `core/parsed_model.py` | `load_visualization_model` (JSON-first) |
| `graph/` | Network plots, bipartite variable–parameter sketch |
| `matrix/` | `MatrixVisualizer`, `extract.py`, `compat.py` (`parse_matrix_data`) |
| `analysis/` | Combined multi-panel plots |
| `plotting/utils.py` | `save_plot_safely`, `safe_tight_layout` |
| `processor.py` | Shim re-exporting core + parse + plotting |

## Visualization Types

| Type | Description |
|------|-------------|
| Network | Spring layout; undirected edges without arrows |
| Matrix | Heatmaps, 3D transition tensors, POMDP analysis |
| Combined | Four-panel summary + standalone panels |
| Bipartite | Variables vs parameter names (name match edges) |

## Tips for AI Assistants

1. **Step 8:** `8_visualization.py` calls `process_visualization` only.
2. **Dependencies:** matplotlib, networkx; optional plotly, seaborn.
3. **Output:** `output/8_visualization_output/` by default.
4. **Stale JSON:** If `*_parsed.json` is older than the source `.md`, a `*_viz_source_note.txt` is written.
5. **Manifest:** Each model folder gets `{model}_viz_manifest.json` listing artifacts, counts, and `_viz_meta` (JSON vs markdown source).

---

**Version:** 1.1.3 | **Step:** 8 (Visualization)
