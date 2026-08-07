# RxInfer.jl Render Module

This module renders validated GNN POMDP specifications to executable RxInfer.jl scripts.

## Public Surface

- `render_gnn_to_rxinfer(gnn_spec, output_path, options=None)`
- Step 11 calls this renderer through `render.pomdp_processor.POMDPRenderProcessor`.
- `render_gnn_spec(..., "rxinfer", ...)` routes to the same canonical renderer for POMDP specs.

## Contract

The renderer consumes `canonical_pomdp_v1` data:

- `A`: observation likelihood, shape `(observation, state)`
- `B`: transition tensor, shape `(next_state, previous_state, action)`
- `C`: observation preferences
- `D`: initial hidden-state prior
- `E`: policy prior when present
- matrix provenance and runtime metadata

Generated scripts import RxInfer.jl and write `simulation_results.json` with schema `rxinfer_simulation_v1`. The canonical renderer (`rxinfer_renderer.py`) dispatches by detected `ModelKind` to per-kind strategies (`model_strategies.py`), each emitting a genuine Julia script that runs `infer()` with `free_energy = true`:

| ModelKind | Strategy | Generated model |
|---|---|---|
| FLAT | `FlatStrategy` | `pomdp_model` — batch smoothing, or per-timestep filtering when `inference_mode: online` is declared in ModelParameters (or passed as a render option) |
| HIERARCHICAL | `HierarchicalStrategy` | `hierarchical_pomdp_model` for two-level exemplars (context latent coupled into the fast prior, mean-field constraints + initialization); 3+ declared levels render as the documented joint composition |
| FACTORED | `FactoredStrategy` | `factored_pomdp_model` — native mean-field two-factor model with multi-parent likelihood (`DiscreteTransition(s1, A_m0, s2)`), per-factor posteriors |
| CONTINUOUS | `ContinuousStrategy` | `continuous_pomdp_model` — linear-Gaussian state space (F/H/Q/R + Gaussian prior from InitialParameterization); beliefs are posterior means, sign-agnostic VFE validation |
| LEARNING | `LearningStrategy` | `learning_pomdp_model` — likelihood matrix A learned as a latent `DirichletCollection` from `dirichlet_A` pseudo-counts; reports learned-A mean and prior/posterior distance to the true A |
| MULTI_AGENT | `MultiAgentStrategy` | joint composition with true kind stamped; per-agent marginals recovered downstream from the `state_factors` echo |

The legacy TOML-based `toml_generator.py` is **deprecated** (`render_gnn_to_rxinfer_toml` emits a `DeprecationWarning`).

## Generated-script outputs

Each rendered `<model>_rxinfer.jl` script emits a required result file and up to
two families of *best-effort* artifacts when the surrounding tooling is present.
The required payload is always written; the optional artifacts are guarded so that
a missing plotting/logging dependency can **never** cause execution failure.

### Required — `simulation_results.json`

The script always writes `simulation_results.json` in the execution working
directory, serialized with schema `rxinfer_simulation_v1`:

- `schema_version`, `success`, `framework`, `model_name`, `num_timesteps`
- `observations_by_modality`, `hidden_states_by_factor`, `actions_by_control_factor`
- `beliefs_by_factor`, `expected_free_energy`, `efe_per_action`, `policy_posterior`
- `observations`, `true_states`, `actions`, `beliefs`
- `variational_free_energy` (genuine VFE trace from `infer()` `free_energy`; previously `Float64[]`)
- `model_parameters` (matrix shapes and dimensions)
- `matrix_provenance` and `runtime_metadata` (seed, script SHA256, `uses_real_rxinfer: true`, schema version, RxInfer/Julia versions)
- `validation` (belief validity, normalisation, action range) and `metrics`

Step 12 `main()` returns a non-zero exit code if `validation.all_valid` is false, so
invalid inference is surfaced without suppressing the result payload itself.

### Optional — structured simulation log

When enabled, the rendering emits structured runtime tracing alongside the result:

```text
simulation.log          # human-readable chronological trace
simulation_log.json     # machine-readable structured events
```

These capture inference progress, per-step belief updates, expected-free-energy
values, and the validation outcome. They are **best-effort**: if logging is disabled
or the target cannot be opened, the script warns and continues writing
`simulation_results.json` normally.

### Optional — Julia-native Plots PNGs

When Julia Plots rendering is available, the script may additionally emit native
`Plots.jl` figures:

```text
belief_evolution.png    # belief posteriors over time
efe_over_time.png       # expected free energy over time
policy_posterior.png    # softmax policy posterior over actions
```

These are **best-effort** and optional — absence of `Plots.jl` (or any plotting
failure) is caught and logged, never propagated as a run failure. They complement
(and do not replace) the Step-16 matplotlib analysis PNGs described in
[`src/analysis/rxinfer/README.md`](../../analysis/rxinfer/README.md).

## Outputs

```text
output/11_render_output/<model>/rxinfer/
├── <model>_rxinfer.jl
└── README.md
```

Step 12 collects runtime outputs into:

```text
output/12_execute_output/<model>/rxinfer/simulation_data/simulation_results.json
```

Guarded best-effort artifacts (`simulation.log`, `simulation_log.json`, and the
Julia-native Plots PNGs) are collected into the same
`output/12_execute_output/<model>/rxinfer/` tree when present.

## Verification

```bash
julia --startup-file=no --project=src/execute/rxinfer -e 'using RxInfer, JSON, Distributions, StatsBase'
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
```
