# Visualization Core Sub-module

## Overview

Core orchestration logic for Step 8 (Visualization). Handles JSON-first model loading from Step 3 parsed output and coordinates graph, matrix, bipartite, and combined-analysis visualization generation.

## Architecture

```
core/
├── __init__.py          # Package exports
├── process.py           # Step-8 visualization orchestration (decomposed helpers)
├── parsed_model.py      # JSON-first model loading and staleness tracking
└── sampling.py          # Pure downsampling for very large models
```

## Key Functions

### process.py

- **`process_visualization(target_dir, output_dir, verbose=False, *, logger=None, **kwargs) -> bool | int`** — Main orchestration entry point called by `8_visualization.py`. Discovers GNN files, loads parsed JSON models, dispatches to graph/matrix/combined visualizers, and writes `visualization_summary.json`. Accepts an optional injected `logger` (the pipeline passes its configured step logger); when omitted, the module-level `"visualization"` logger is used (direct-call behavior preserved). Returns `True`, warning code `2`, or `False`.
- **`process_single_gnn_file(gnn_file, results_dir, verbose=False) -> List[str]`** — Per-file pipeline: cache check → model load → sampling → network/bipartite/matrix/combined rendering → manifest.
- **`discover_visualization_files(target_dir, recursive=True) -> List[Path]`** — Deterministic `*.md` / `*.gnn` discovery honoring the `recursive` flag.
- **`load_cached_artifacts(model_dir, source_mtime) -> List[str]`** — mtime-gated PNG cache reuse; clears stale cache entries.
- **`render_matrix_artifacts(matrices, model_dir, model_name, visualizer, verbose=False) -> List[str]`** — 2D heatmap vs 3D tensor / three.js / POMDP-analysis render dispatch.
- **`write_viz_manifest(model_name, parsed_data, artifacts, model_dir) -> Optional[Path]`** — Writes `{model}_viz_manifest.json`.
- **`write_sampling_note(model_dir, model_name, summary) -> None`** — Writes `{model}_sampling_note.txt`.

### parsed_model.py

- **`load_visualization_model(gnn_file, content, results_dir, verbose) -> Dict`** — JSON-first model loader: prefers `{model}_parsed.json` from Step 3 output; falls back to `parse/markdown.py` raw parsing.
- **`resolve_gnn_step3_output_dir(results_dir) -> Path`** — Locates the Step 3 output directory for JSON model files.
- **`write_stale_json_note_if_needed(gnn_file, parsed_json, output_dir)`** — Writes `*_viz_source_note.txt` when the source `.md` is newer than the parsed JSON.

### sampling.py (pure — no plotting imports)

- **`sample_parsed_data(parsed_data, variable_limit=100, matrix_limit=5) -> bool`** — Downsampling for very large models: truncates variables, filters connections to surviving endpoints, caps matrices, records `_sampling_applied` counts. Returns `True` when applied.
- **`SamplingSummary`** — TypedDict for the before/after counts.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 3.2.0
**Last Updated**: 2026-09-04
