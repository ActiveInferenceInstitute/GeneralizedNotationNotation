# POMDP GridWorld Fixtures

Maintained 3x3 discrete GridWorld POMDP fixture for cross-framework validation.

The model uses 9 hidden states, 9 observations, and 5 actions: `up`, `down`, `left`, `right`, and `stay`. Transitions are boundary-clamped, observations are noisy identity emissions, the agent starts at the upper-left cell, and preferences favor the lower-right goal observation.

The fixture is intended for strict render, execute, analysis, and visualization checks across PyMDP, RxInfer.jl, and ActiveInference.jl.

```bash
uv run pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
