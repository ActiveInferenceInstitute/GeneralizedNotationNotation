# RxInfer Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for RxInfer.jl simulation results. Part of the Analysis module (Step 16). Consumes the genuine `@model` + `infer()` pipeline outputs: `rxinfer_simulation_v1` with real smoothed posteriors and a `variational_free_energy` trace populated with per-iteration VFE values (length = INFERENCE_ITERATIONS), enabling real convergence and free-energy analysis. Validation includes `inference_converged`, `vfe_present`, and `belief_entropy_ok`.

## Module Structure

```
analysis/rxinfer/
├── __init__.py          # Public API
├── analyzer.py          # Analysis from execution logs + convergence diagnostics + per-factor beliefs
├── animator.py          # Animated HTML visualizations
├── gif_animator.py      # Publication-style GIF animations + reproducibility manifest sidecar
├── dashboard.py         # Interactive HTML dashboard over the GIF batch (roadmap A5)
├── cross_framework.py   # Cross-framework comparison (roadmap A6)
├── README.md            # Human documentation
└── AGENTS.md            # This file
```

### gif_animator.py

`generate_gif_animation(results, output_path, ...)` renders the 2×3
publication-style (white) GIF: beliefs, states, Bayesian graph model,
VFE, EFE-per-action heatmap (D6), and policy-posterior stackplot (D8).
For multi-factor results (`model_parameters.state_factors` with >1
size>1 factor) the top-left joint-belief panel becomes per-factor
marginal small-multiples (D4). Every GIF writes a `.manifest.json`
sidecar (A7: spec hash, Julia/RxInfer versions, seed, timesteps,
iterations, belief accuracy).

### dashboard.py

`generate_dashboard(animations_dir, output_path)` builds a single
self-contained HTML page over all GIFs + manifests with category
grouping and filtering.

### analyzer.py additions

- `compute_per_factor_beliefs(data)` — un-flattens joint posteriors into
  per-factor marginals using the `state_factors` echo in results JSON
  (C-order reshape; returns `{}` for flat models and for artifacts written
  before the echo existed, raises `ValueError` on inconsistent factor sizes).
- `_compute_convergence_diagnostics(...)` — VFE slope, convergence rate,
  iterations-to-convergence (D5), plotted alongside free energy.

### cross_framework.py

Implements roadmap **A6**: renders one GNN file to RxInfer.jl, PyMDP, and
ActiveInference.jl from a single parsed spec, executes each, and emits a
self-contained HTML comparison.

- `run_cross_framework_comparison(gnn_file, output_dir) -> str` — entry point;
  raises `FileNotFoundError` for a missing GNN file.
- `render_comparison_html(model_name, runs, output_path) -> str` — pure
  renderer over `FrameworkRun` records, unit-testable without Julia.
- `FrameworkRun` — dataclass carrying `framework`, `status`
  (`success` / `validation_failed` / `render_failed` / `execution_failed` /
  `unavailable` / `invalid_results`), `detail`, and optional `results`.

Exit-code contract: only exit 0 with `simulation_results.json` is a clean
success; exit 1 with results is kept and flagged as `validation_failed`;
anything else is `execution_failed` with the stderr tail logged at error
level. PyMDP results are redirected into the per-framework directory via
`PYMDP_OUTPUT_DIR`; both Julia backends run under their committed
`--project` environments resolved relative to this file, not the CWD.

## Key Functions

### analyzer.py

- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point
- `_parse_rxinfer_outputs(filepath)` - Parse RxInfer outputs
- `_analyze_messages(data)` - Message flow analysis
- `_analyze_convergence(data)` - Convergence tracking
- `_generate_report(metrics)` - Report generation

## Integration Points

**Upstream:** Execute module (Step 12) produces RxInfer simulation results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

- pathlib, json, logging: Core Python
- numpy (optional): Numerical operations
- matplotlib (optional): Visualization

---

**Version:** 3.0.0
**Last Updated:** 2026-01-23
