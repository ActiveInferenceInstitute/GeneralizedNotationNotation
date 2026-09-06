# Visualization Tests

Pytest coverage for `src/gnn/visualization/`.

This folder contains module-focused tests for graph, matrix, Mermaid, D2, and artifact generation behavior.

## Test Files

- `test_d2_visualizer.py` — D2 visualization integration: diagram generation, CLI absence handling, and artifact output.
- `test_mermaid_converter.py` — GNN-to-Mermaid conversion (node shape inference, edge formatting, output rendering).
- `test_mermaid_parser.py` — Mermaid-to-GNN conversion (metadata extraction, structure reconstruction).
- `test_threejs_tensor_explorer.py` — ThreeJS tensor explorer HTML/JSON output and the CDN-blocked JSON data path.
- `test_visualization_artifacts.py` — step-8 sidecars: network stats orientation, ontology legend, and viz manifest.
- `test_visualization_backends.py` — backend status reporting and the theme single-source-of-truth invariant.
- `test_visualization_comprehensive.py` — real-data visualization generation with matplotlib backend handling, progress tracking, and error recovery.
- `test_visualization_matrices.py` — the `MatrixVisualizer` class and matrix-specific visualizations.
- `test_visualization_matrix_collect.py` — `gnn.visualization.matrix.extract.collect_visualization_matrices`.
- `test_visualization_matrix_compat.py` — the `matrix_compat` forwarding surface to `gnn.visualization.matrix.compat`.
- `test_visualization_module_info.py` — `gnn/visualization/__init__.py` public helpers and the exported public surface.
- `test_visualization_ontology.py` — ontology visualization behavior.
- `test_visualization_overall.py` — module-level aggregate contract for the visualization folder.
- `test_visualization_pkg_api.py` — package-root public API surface (module README "Public API" table) and the injected-logger contract.
- `test_visualization_sampling.py` — pure downsampling helpers in `gnn.visualization.core.sampling`.
- `test_visualization_stats.py` — `gnn.visualization.graph.stats.compute_connection_statistics`.
- `test_visualization_theme.py` — colour-palette and edge-style lookup helpers plus the module-level constants that act as the single source of truth for Steps 8/9 rendering.

Run:

```bash
uv run --extra dev python -m pytest tests/visualization/ -q
```
