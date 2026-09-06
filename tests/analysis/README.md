# Analysis Tests

Pytest coverage for `src/gnn/analysis/`.

This folder contains module-focused tests for analysis processing, empty-input handling, and result extraction.

## Test Files

- `test_active_inference_math.py` — unit tests for the Active Inference mathematical functions (entropy, KL divergence, free energies, information gain) in `src/gnn/analysis/post_simulation.py`.
- `test_analysis_empty_input.py` — empty-input handling for analysis processors.
- `test_analysis_extraction.py` — result extraction logic.
- `test_analysis_mcp_wrappers.py` — MCP wrapper behavior for the analysis module.
- `test_analysis_overall.py` — module-level aggregate contract for the analysis folder.
- `test_analysis_post_simulation.py` — post-simulation processing.
- `test_analysis_viz_base.py` — visualization base helpers.
- `test_flat_payload_analyzer.py` — flat payload analysis.
- `test_framework_common.py` — shared analysis-framework helpers.
- `test_generate_cross_model_report.py` — cross-model report generation.
- `test_interpretability_summary.py` — interpretability summary output.
- `test_math_utils_edge.py` — edge cases for analysis math utilities.
- `test_numpyro_pytorch_analyzers.py` — NumPyro/PyTorch analyzer behavior.
- `test_rxinfer_analyzer_comprehensive.py` — comprehensive RxInfer analyzer coverage.
- `test_rxinfer_animator.py` — RxInfer animation output.
- `test_rxinfer_cross_framework.py` — cross-framework analysis behavior.
- `test_rxinfer_dashboard.py` — RxInfer dashboard rendering.
- `test_rxinfer_gif_animator.py` — RxInfer GIF animation output.

Run:

```bash
uv run --extra dev python -m pytest tests/analysis/ -q
```
