# Stan Renderer (`src/render/stan/`)

## Files

| File | Role |
|---|---|
| `stan_renderer.py` | `render_gnn_to_stan(gnn_spec, output_path, options)` — writes `<stem>.stan` + `<stem>.py` driver; branches on `render.continuous_common.is_continuous_spec` |
| `__init__.py` | exports `render_gnn_to_stan`, legacy `render_stan` |

## Contract

- Called from `render.pomdp_processor.POMDPRenderProcessor._call_stan_renderer`
  with the canonical spec (`initialparameterization` A/B/C/D for discrete
  models; `F/H/Q/R/prior_mean/prior_cov[/goal_mean/control_gain]` for
  continuous models, `model_kind == "continuous"`).
- `output_path` is the **driver** (`*_stan.py`); the `.stan` program shares
  its stem. Both paths are returned as artifacts so Step 12's manifest-based
  discovery executes the driver.
- Discrete: `B` must be `(next, prev, action)`; 2-D `B` is promoted to one
  action. `T = max(2, num_timesteps)`.
- Continuous: shapes validated by `render.continuous_common.extract_continuous_spec`.

## Invariants

- Generated Stan must compile under CmdStan ≥ 2.33 (`array[...]` syntax,
  vectorised matrix ops, `multi_normal_lpdf`).
- Discrete inference budget: NUTS when `S·O ≤ stan_nuts_param_budget`
  (default 1024), else L-BFGS MAP — never an unbounded sampler on a
  multi-agent joint composition.
- No fabricated data: the driver simulates from the declared model with
  `ModelParameters.random_seed`; posterior summaries are computed from the
  actual draws (`fit.stan_variables()`), R-hat from `fit.summary()`.

## Tests

`src/tests/render/test_render_stan.py` — legacy sketch tests plus discrete and
continuous program generation; compile checks run only when CmdStan is
installed.
