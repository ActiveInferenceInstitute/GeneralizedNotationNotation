# Visualization Core

Core processing engine for GNN visualization: model loading, file processing, and output directory resolution.

## Exports

- `load_visualization_model()` — Load a parsed GNN model for visualization
- `process_single_gnn_file()` — Generate all visualizations for a single GNN file
- `process_visualization()` — Pipeline entry point for step 8
- `resolve_gnn_step3_output_dir()` — Locate step 3 output for cross-step data access

## Dependencies

- `visualization.parse` for GNN content parsing
- `visualization.graph` and `visualization.plotting` for rendering
