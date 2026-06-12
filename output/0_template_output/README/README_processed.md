
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/pomdp_gridworld/README.md
# Processed on: 2026-05-22T06:10:26.461854
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# POMDP GridWorld Fixtures

Maintained 3x3 discrete GridWorld POMDP fixture for cross-framework validation.

The model uses 9 hidden states, 9 observations, and 5 actions: `up`, `down`, `left`, `right`, and `stay`. Transitions are boundary-clamped, observations are noisy identity emissions, the agent starts at the upper-left cell, and preferences favor the lower-right goal observation.

The fixture is intended for strict render, execute, analysis, and visualization checks across PyMDP, RxInfer.jl, and ActiveInference.jl.

The streamlined command below emits one current-schema `simulation_results.json` per
framework, statistical analysis, per-framework PNG plots, per-framework belief and
3x3 state trajectory GIFs, and a cross-framework GridWorld GIF plus manifest under
`output/16_analysis_output/cross_framework/`.

```bash
uv run pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```

```bash
uv run python src/main.py --only-steps "3,5,8,11,12,16" --target-dir input/gnn_files/pomdp_gridworld --frameworks "pymdp,rxinfer,activeinference_jl" --verbose
```

