# RxInfer Renderer — Technical Specification

**Version**: 1.6.0

## Purpose

Generates Julia code using the RxInfer.jl reactive message-passing framework.

## Code Generation

- Maps GNN model structure to RxInfer factor graph specification
- Generates probabilistic model definition and inference queries
- Produces TOML configuration for model parameters

## Output

- Julia script files (`.jl`) using RxInfer API
- TOML parameter files

## Generated-Script Output Contract

Each rendered `.jl` script emits one required artifact plus optional guarded outputs.

### Required
- `simulation_results.json` — always written, schema `rxinfer_simulation_v1`
  (observations, hidden states, actions, beliefs by factor, expected free energy,
  policy posterior, matrix provenance, runtime metadata, validation, metrics).

### Optional (best-effort, never cause execution failure)
- `simulation.log` — human-readable chronological trace.
- `simulation_log.json` — machine-readable structured runtime events.
- `belief_evolution.png`, `efe_over_time.png`, `policy_posterior.png` — Julia-native
  `Plots.jl` figures emitted only when Plots rendering is available.

All optional artifacts are guarded: a missing dependency, failed write, or plotting
error is caught and logged, and the script continues to write `simulation_results.json`
normally. Step 12 `main()` returns non-zero only if `validation.all_valid` is false.

## Architecture

```
rxinfer/
├── rxinfer_renderer.py     # Core renderer (625 lines)
├── toml_generator.py       # TOML config generation (995 lines)
└── ...
```

## Dependencies

Target: `julia >= 1.8`, `RxInfer.jl >= 3.0`
