# PyMDP Analysis Module

Step 16 PyMDP analysis consumes current Step 12 PyMDP execution output and
writes framework-specific plots.

## Public API

Exported from `src/analysis/pymdp/__init__.py`:

- `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`
- `PyMDPVisualizer`
- `create_visualizer(...)`
- `save_all_visualizations(...)`

## Input Contract

The only accepted PyMDP execution schema is `pymdp_simulation_v1` in
`simulation_results.json`. Required fields include observations by modality,
hidden states by factor, actions by control factor, beliefs by factor, expected
and variational free energy traces, policy posterior, validation, matrix
provenance, and runtime metadata.

Older flat PyMDP result shapes are rejected with diagnostics rather than
recovered into plots.

## Outputs

Analysis writes files under `output/16_analysis_output/pymdp/<model_slug>/`,
including belief, action, free-energy, observation, and preference plots when
the corresponding schema fields are present.

## Verification

```bash
uv run pytest \
    src/tests/analysis/test_analysis_post_simulation.py \
    src/tests/analysis/test_analysis_overall.py \
    -q --tb=short
```
