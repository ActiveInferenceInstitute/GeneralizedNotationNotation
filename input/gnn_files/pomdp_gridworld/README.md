# POMDP GridWorld Fixtures

Maintained 3x3 discrete GridWorld POMDP fixture for cross-framework validation.

The model uses 9 hidden states, 9 observations, and 5 actions: `up`, `down`, `left`, `right`, and `stay`. Transitions are boundary-clamped, observations are noisy identity emissions, the agent starts at the upper-left cell, and preferences favor the lower-right goal observation.

The fixture is intended for strict render, execute, analysis, and visualization checks across PyMDP, RxInfer.jl, and ActiveInference.jl.

The streamlined command below emits one current-schema `simulation_results.json` per
framework, statistical analysis, per-framework PNG plots, per-framework belief and
3x3 state trajectory GIFs, and a cross-framework GridWorld GIF plus manifest under
`output/16_analysis_output/cross_framework/`.

```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```

```bash
uv run --extra dev python src/main.py --only-steps "3,5,8,11,12,16" --target-dir input/gnn_files/pomdp_gridworld --frameworks "pymdp,rxinfer,activeinference_jl" --verbose
```

## Public root-output publication

For a public `output/` refresh, run the full pipeline against this fixture and
all registered render targets:

```bash
uv run --extra dev python src/main.py \
  --target-dir input/gnn_files/pomdp_gridworld \
  --output-dir output \
  --frameworks all \
  --strict \
  --strict-framework-success \
  --enable-round-trip \
  --enable-cross-format \
  --execution-summary-detail \
  --advanced-stats \
  --mcp-strict-validation \
  --verbose
```

Then validate the generated tree before committing:

```bash
uv run --extra dev python scripts/check_pomdp_gridworld_outputs.py output
```

The publication contract requires one parsed GridWorld model, render artifacts
for all registered render targets (`pymdp`, `rxinfer`, `activeinference_jl`,
`jax`, `discopy`, `pytorch`, `numpyro`, `stan`, and `bnlearn`), successful
execution evidence for PyMDP/RxInfer.jl/ActiveInference.jl, explicit
success/skipped/error accounting for optional runtime surfaces, GridWorld
PNG/GIF analysis artifacts, and POMDP-specific report and website pages.
