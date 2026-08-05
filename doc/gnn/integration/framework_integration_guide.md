# GNN Framework Integration Guide

**Version**: 3.0.0 (matches repo root `pyproject.toml`)
**Status**: Maintained

## Pipeline Integration

GNN framework integration is handled by **Steps 11 and 12** of the processing pipeline:

- **`src/11_render.py`** → Code generation for PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX, PyTorch, NumPyro, Stan, bnlearn
  - See: **[src/render/AGENTS.md](../../../src/render/AGENTS.md)** for rendering module details
- **`src/12_execute.py`** → Execution of rendered simulation scripts
  - See: **[src/execute/AGENTS.md](../../../src/execute/AGENTS.md)** for execution module details

**Quick Start:**

```bash
# Generate and execute code for all frameworks
python src/main.py --only-steps "11,12" --target-dir input/gnn_files --verbose

# Execute specific frameworks only
python src/12_execute.py --frameworks "pymdp,jax" --verbose
```

For complete pipeline documentation, see **[src/AGENTS.md](../../../src/AGENTS.md)**.

## Overview

This guide describes how GNN models actually flow through the framework integration
surface. There is no separate framework-agnostic intermediate-representation class in
the codebase — each renderer under `src/render/<framework>/` consumes the parsed GNN
model dictionary directly (as produced by `src/gnn/parsers/`) and emits framework-native
code. Per-framework rendering, execution, and troubleshooting detail lives in the
per-framework guides linked below; this document covers the parts of the pattern that
are shared across frameworks — the render/execute/analyze pipeline shape and the
cross-framework comparison step.

## Supported Frameworks

Step 11 renders to 9 backends (see `src/render/AGENTS.md`):

| Framework | Per-framework guide |
|-----------|----------------------|
| PyMDP | [implementations/pymdp.md](../implementations/pymdp.md) |
| RxInfer.jl | [implementations/rxinfer.md](../implementations/rxinfer.md) |
| ActiveInference.jl | [implementations/activeinference_jl.md](../implementations/activeinference_jl.md) |
| JAX | [implementations/jax.md](../implementations/jax.md) |
| DisCoPy | [implementations/discopy.md](../implementations/discopy.md) |
| PyTorch | [implementations/pytorch.md](../implementations/pytorch.md) |
| NumPyro | [implementations/numpyro.md](../implementations/numpyro.md) |
| Stan | [implementations/stan.md](../implementations/stan.md) |
| bnlearn | `src/render/AGENTS.md` (no standalone guide yet) |

Use `--frameworks` (plural) on `src/11_render.py` / `src/12_execute.py` to restrict to a
subset, e.g. `--frameworks pymdp,jax`, or `--frameworks lite` for the `pymdp,jax,discopy,bnlearn`
quick subset. There is no per-framework singular `--framework` flag.

## Render → Execute → Analyze Pipeline per Framework

The GNN pipeline processes each framework through three stages. This section documents
framework-specific behavior at each stage.

### Pipeline Overview per Framework

| Framework | Render Target | Script Type | Executor | Data Extractor | Analysis Metrics |
|-----------|---------------|-------------|----------|----------------|------------------|
| **PyMDP** | `render/pymdp/` | `.py` | Python `subprocess` | `extract_pymdp_data()` | Beliefs, actions, free energy, observations |
| **RxInfer.jl** | `render/rxinfer/` | `.jl` | Julia `subprocess` | `extract_rxinfer_data()` | Posterior distributions + genuine VFE |
| **ActiveInference.jl** | `render/activeinference_jl/` | `.jl` | Julia `subprocess` | `extract_activeinference_jl_data()` | Full Active Inference fields from CSV |
| **JAX** | `render/jax/` | `.py` | Python `subprocess` | — | GPU-accelerated simulation output |
| **DisCoPy** | `render/discopy/` | `.py` | Python `subprocess` | `extract_discopy_data()` | Diagram executions, categorical outputs |

`extract_pymdp_data`, `extract_rxinfer_data`, `extract_activeinference_jl_data`, and
`extract_discopy_data` live in `src/execute/data_extractors.py` (raw stdout/stderr
parsing) and `src/analysis/framework_extractors.py` (post-simulation JSON-payload
normalization) — see those modules for the current field-level schema.

### PyMDP Pipeline Details

**Render:** `src/render/pymdp/` generates complete PyMDP Python scripts with A, B, C, D
matrices, `Agent` instantiation, a simulation loop, and result serialization.

**Execute:** Python subprocess with `PYTHONPATH` extended for PyMDP imports. Dependency
check: `import pymdp`.

**Analyze:** `extract_pymdp_data()` reads beliefs, actions, free energy, and observations
from JSON output. Supports reading from collected files in `output/pymdp_simulations/`.

### ActiveInference.jl Pipeline Details

**Render:** Generates Julia scripts using `ActiveInference.jl` with POMDP agent setup,
environment initialization, and a simulation loop. Creates companion `config.toml` files.

**Execute:** Julia subprocess with a package-availability check for `ActiveInference`,
`GraphPPL`. Reads output from `simulation_results.csv`.

**Analyze:** `extract_activeinference_jl_data()` performs CSV parsing to extract beliefs,
actions, free energy, observations, states, and policies, with numerical parsing and
error handling.

### RxInfer.jl Pipeline Details

**Render:** `src/render/rxinfer/rxinfer_renderer.py` emits a genuine Julia script per
model defining `@model function pomdp_model(y, A, B, D, u, T)` with `Categorical`
and `DiscreteTransition` nodes (the legacy TOML-based `toml_generator.py` is
deprecated). The script runs `infer()` with `free_energy = true`.

**Execute:** Julia subprocess invoking
`julia --startup-file=no --project=src/execute/rxinfer <script>`. The committed
`Project.toml` + `Manifest.toml` under `src/execute/rxinfer/` pin RxInfer 5.5.0;
`setup_environment.jl` activates and instantiates it (`Pkg.activate()` +
`Pkg.instantiate()`, no runtime `Pkg.add`). Each run records `Random.seed!(seed)`
and the script SHA256 in `runtime_metadata`.

**Analyze:** `extract_rxinfer_data()` reads posterior distributions plus the genuine
`variational_free_energy` trace from `rxinfer_simulation_v1`
(`simulation_data/simulation_results.json`).

### JAX Pipeline Details

**Render:** Generates JAX-based Python scripts with JIT compilation, GPU acceleration,
and automatic differentiation. Supports POMDP simulation with vectorized operations.

**Execute:** Python subprocess, same as PyMDP but without the PyMDP-specific dependency
check.

**Analyze:** Reads simulation output from `jax_outputs_*/` directories.

### DisCoPy Pipeline Details

**Render:** Generates categorical string diagrams using DisCoPy's rigid category
framework. Creates morphisms for state-observation connections and composes them via
sequential (`>>`) or tensor (`@`) products.

**Execute:** Python subprocess. Diagram evaluation produces categorical composition
results.

**Analyze:** `extract_discopy_data()` counts diagram executions and extracts categorical
outputs from `discopy_diagrams/`. Treats individual diagram evaluations as simulation
"steps".

### Cross-Framework Analysis

After individual framework execution, Step 16 (`src/analysis/analyzer.py`) and Step 23
(`src/report/`) perform comparative analysis and dashboard generation:

1. **`analyze_framework_outputs()`** — loads and normalizes results from all frameworks
   into standard JSON targets.
2. **`generate_framework_comparison_report()`** — generates comparison metrics (execution
   time, convergence, accuracy).
3. **`visualize_cross_framework_metrics()`** — native side-by-side metric visualizations.
4. **`generate_unified_framework_dashboard()`** (`src/analysis/visualizations.py`) —
   generates an interactive HTML/D3.js dashboard comparing beliefs, action
   distributions, and observation trajectories in the browser.

## Related Documentation

- **[PyMDP Integration Guide](../implementations/pymdp.md)**: Detailed PyMDP-specific documentation
- **[RxInfer Integration Guide](../implementations/rxinfer.md)**: RxInfer.jl integration
- **[DisCoPy Integration Guide](../implementations/discopy.md)**: Categorical diagram processing
- **[Troubleshooting Guide](../operations/gnn_troubleshooting.md)**: Error and troubleshooting reference

---

**Integration Guide Version**: 3.0.0
**Framework Coverage**: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn
**Status**: Maintained
