# PyMDP Analysis Agent Guide

## Purpose

Analyze PyMDP Step 12 results after execution has produced
`pymdp_simulation_v1` JSON artifacts. This folder owns analysis and plotting
only; it does not run PyMDP simulations.

## Public Surface

Use `src/analysis/pymdp/__init__.py` as source of truth:

- `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`
- `PyMDPVisualizer`
- `create_visualizer(...)`
- `save_all_visualizations(...)`

## Contract

- Input must be `simulation_results.json` with
  `"schema_version": "pymdp_simulation_v1"`.
- Analyzer discovery prefers Step 12 execution summaries and collected
  `pymdp/simulation_data` outputs.
- Unsupported PyMDP schemas are skipped with an error log.
- Plots are written under `output/16_analysis_output/pymdp/<model_slug>/`.

## Testing

- PyMDP analysis schema extraction:
  `src/tests/analysis/test_analysis_post_simulation.py`
- Pipeline analysis integration:
  `src/tests/analysis/test_analysis_overall.py`
- PyMDP execution contracts that produce the input schema:
  `src/tests/execute/test_pymdp_contracts.py`

**Version:** 1.6.0
**Last Updated:** 2026-05-14
