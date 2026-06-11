# Visualization Core Sub-module

## Overview

Core orchestration logic for Step 8 (Visualization). Handles JSON-first model loading from Step 3 parsed output and coordinates graph, matrix, bipartite, and combined-analysis visualization generation.

## Architecture

```
core/
├── __init__.py          # Package exports
├── process.py           # Step-8 visualization orchestration
└── parsed_model.py      # JSON-first model loading and staleness tracking
```

## Key Functions

### process.py

- **`process_visualization(target_dir, output_dir, verbose=False, **kwargs) -> bool`** — Main orchestration entry point called by `8_visualization.py`. Discovers GNN files, loads parsed JSON models, dispatches to graph/matrix/combined visualizers, and writes `visualization_summary.json`.
- **`process_single_gnn_file(gnn_file, results_dir, output_dir, verbose) -> Dict`** — Per-file processing: loads model, generates network graphs, matrix heatmaps, bipartite diagrams, combined analysis panels, and writes per-model manifest.

### parsed_model.py

- **`load_visualization_model(gnn_file, content, results_dir, verbose) -> Dict`** — JSON-first model loader: prefers `{model}_parsed.json` from Step 3 output; falls back to `parse/markdown.py` raw parsing.
- **`resolve_gnn_step3_output_dir(results_dir) -> Path`** — Locates the Step 3 output directory for JSON model files.
- **`write_stale_json_note_if_needed(gnn_file, parsed_json, output_dir)`** — Writes `*_viz_source_note.txt` when the source `.md` is newer than the parsed JSON.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
