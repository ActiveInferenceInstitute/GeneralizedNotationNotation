# Advanced Visualization Tests

Pytest coverage for `src/gnn/advanced_visualization/`.

This folder contains module-focused tests for advanced visualization behavior and generated artifacts.

## Test Files

- `test_advanced_visualization_composability.py` — composability helpers: `_shared.record_attempt` aggregate bookkeeping and D2-CLI absence message filtering.
- `test_advanced_visualization_html_generator.py` — `HTMLVisualizationGenerator` success and error paths against real structured model data.
- `test_advanced_visualization_interactive.py` — `_generate_interactive_plotly_dashboard` against live Plotly and numpy.
- `test_advanced_visualization_overall.py` — module-level aggregate contract: D2 diagram generation, dashboards, and interactive visualizations.
- `test_advanced_visualization_polish.py` — polish-pass seams: D2 `parsed_json_dir`, network-graph indices, extractor dependency injection, and theme parity.
- `test_advanced_visualization_public_api.py` — public API surface: `get_module_info`, `FEATURES`, `__version__`, `create_dashboard_section`, `create_visualization_from_data`, `create_heatmap_visualization`, `create_timeline_visualization`, `D2DiagramSpec`, `D2GenerationResult`.
- `test_advanced_visualization_public_api_refactor.py` — the additive public-API surface: `VIZ_TYPE_CHOICES` and the live `probe_capabilities()` environment probe.
- `test_advanced_visualization_shared.py` — `gnn/advanced_visualization/_shared.py`: `normalize_connection_format`, `validate_visualization_data`.
- `test_advanced_visualization_statistical.py` — `_generate_statistical_plots` and `_generate_matrix_correlations` against the Agg matplotlib backend.

Run:

```bash
uv run --extra dev python -m pytest tests/advanced_visualization/ -q
```
