# Specification: Rxinfer

## Real RxInfer.jl Integration
The RxInfer.jl surface is a genuine `@model` + `infer()` variational
message-passing integration:

- The canonical renderer (`src/gnn/render/rxinfer/rxinfer_renderer.py`) emits a
  Julia script per exemplar model defining
  `@model function pomdp_model(y, A, B, D, u, T)` with `Categorical` and
  `DiscreteTransition` nodes, solved by `infer()` with `free_energy = true`.
- `variational_free_energy` is populated with genuine values (previously
  `Float64[]`); EFE and policy selection remain custom logic.
- Execution uses a committed `Project.toml` + `Manifest.toml` pinning
  RxInfer 6.1.0 under `src/gnn/execute/rxinfer/`, invoked via
  `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`.
  `setup_environment.jl` uses `Pkg.activate()` + `Pkg.instantiate()` — no
  runtime `Pkg.add`.
- Model-kind and inference admission are explicit in the
  [renderer registry](../../src/gnn/render/framework_registry.py) and
  [RxInfer guide](README.md). Current exemplar selection is owned by the
  [input index](../../input/gnn_files/INDEX.md); render success is separate
  from exact-source native numerical acceptance.

## Documentation contract

[README.md](README.md) owns the topic entrypoints; [AGENTS.md](AGENTS.md) guides
maintenance. Follow the [documentation rules](../../.agent_rules/documentation.md) for links, executable
examples and source-bound claims. Research scripts and quoted upstream code have
their own prerequisites; their presence does not register a pipeline backend.
