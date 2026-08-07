# Specification: Rxinfer

## Design Requirements
This module (`rxinfer`) maps structural logic to the overall execution graph.
It ensures that `Rxinfer` tasks resolve without runtime dependency loops.

## Real RxInfer.jl Integration
The RxInfer.jl surface is a genuine `@model` + `infer()` variational
message-passing integration:

- The canonical renderer (`src/render/rxinfer/rxinfer_renderer.py`) emits a
  Julia script per exemplar model defining
  `@model function pomdp_model(y, A, B, D, u, T)` with `Categorical` and
  `DiscreteTransition` nodes, solved by `infer()` with `free_energy = true`.
- `variational_free_energy` is populated with genuine values (previously
  `Float64[]`); EFE and policy selection remain custom logic.
- Execution uses a committed `Project.toml` + `Manifest.toml` pinning
  RxInfer 5.5.0 under `src/execute/rxinfer/`, invoked via
  `julia --startup-file=no --project=src/execute/rxinfer <script>`.
  `setup_environment.jl` uses `Pkg.activate()` + `Pkg.instantiate()` — no
  runtime `Pkg.add`.
- All 29 exemplar GNN files render and execute successfully. Step 6 validation
  includes `inference_converged` and `vfe_present`.

## Components
Expected available types: No specific classes exported.
