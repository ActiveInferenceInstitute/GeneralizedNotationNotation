# RxInfer.jl Integration for GNN

> **📋 Document Metadata**  
> **Type**: Framework Integration Guide | **Audience**: Researchers, Developers | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [GNN RxInfer Guide](gnn_rxinfer.md) | [Framework Integration](../gnn/integration/framework_integration_guide.md) | [Main Documentation](../README.md) | [doc/SPEC.md](../SPEC.md) (versioning)

## Overview

This directory contains documentation, scripts, and resources for integrating GNN (Generalized Notation Notation) models with **RxInfer.jl**, a Julia-based Bayesian inference framework. RxInfer.jl provides genuine ``@model`` + ``infer()`` variational message-passing inference for probabilistic models, making it ideal for Active Inference simulations.

**Status**: ✅ Production Ready  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[gnn_rxinfer.md](gnn_rxinfer.md)**: Complete RxInfer.jl integration guide
- **[Multiagent_GNN_RxInfer.jl](Multiagent_GNN_RxInfer.jl)**: Validation script

### Main Documentation
- **[doc/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Framework Integration](../gnn/integration/framework_integration_guide.md)**: Framework integration overview
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies
- **[Multi-agent Systems](../gnn/advanced/gnn_multiagent.md)**: Multi-agent modeling

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/AGENTS.md](../../src/AGENTS.md)**: Implementation details

## Contents

**Files**: 12+ | **Subdirectories**: 1

### Core Files

- **`gnn_rxinfer.md`**: Complete RxInfer.jl integration guide
  - RxInfer.jl framework overview
  - GNN to RxInfer.jl translation
  - Code generation patterns
  - Example models and usage

- **`Multiagent_GNN_RxInfer.jl`**: Validation script
  - Validates GNN to RxInfer.jl translation
  - Two-stage validation process
  - Configuration file generation testing

- **`engineering_rxinfer_gnn.md`**: Engineering guide
  - Technical implementation details
  - Best practices and patterns

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

### Subdirectories

- **`multiagent_trajectory_planning/`**: Multi-agent trajectory planning examples
  - Complete RxInfer.jl implementations
  - Configuration examples
  - Results and analysis

## RxInfer.jl Integration

### Framework Overview

**RxInfer.jl** is a Bayesian inference framework for Julia that provides genuine
`@model` + `infer()` variational message-passing inference over factor graphs:

- **Genuine `@model` + `infer()` pipeline**: Generative models are defined with
  `@model` using `Categorical` and `DiscreteTransition` nodes and solved with
  `infer()` (`free_energy = true`), returning real posteriors over hidden states
  and genuine variational free energy traces. The default pipeline is **offline
  batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation**;
  an optional **online mode** (`inference_mode: online`) runs `infer()` per
  timestep with the filtered posterior driving action selection. If `infer()`
  fails, the script crashes (no fallback).
- **Variational Message Passing**: Optimized inference algorithms over factor graphs
- **Factor Graph Models**: Natural representation of Active Inference models
- **Reproducible Execution**: A committed Julia environment
  (`Project.toml` + `Manifest.toml` pinning RxInfer 5.5.0 under
  `src/execute/rxinfer/`) with `--project=<env>` execution
- **Multi-agent Support**: Coordinated multi-agent systems

### GNN to RxInfer.jl Translation

The GNN pipeline translates GNN models to RxInfer.jl through:

1. **Model Parsing**: GNN `canonical_pomdp_v1` syntax parsed into structured representation
2. **Factor Graph Construction**: Active Inference components mapped to factor graph
3. **Code Generation**: Julia code generation with RxInfer.jl `@model` + `infer()` API
   (emitted by `src/render/rxinfer/rxinfer_renderer.py`)
4. **Environment Setup**: Committed `Project.toml` + `Manifest.toml` under
   `src/execute/rxinfer/` pins RxInfer 5.5.0; `setup_environment.jl` uses
   `Pkg.activate()` + `Pkg.instantiate()` (no runtime `Pkg.add`)
5. **Validation**: Automated validation of generated code and inference results

### Validation Process

The pipeline validates end-to-end render and execution for every exemplar GNN
model:

#### Render (Step 11)
- Discovers all 29 exemplar GNN files under `input/gnn_files/**` and emits an
  executable `*_rxinfer.jl` script per model via `rxinfer_renderer.py`

#### Execute (Step 12)
- Runs each rendered `.jl` with
  `julia --startup-file=no --project=src/execute/rxinfer <script>`
- `setup_environment.jl` activates and instantiates the committed environment
  (`Pkg.activate()` + `Pkg.instantiate()`, no runtime `Pkg.add`)
- Records `random.seed!(seed)` and the script SHA256 in `runtime_metadata`
  for byte-identical, reproducible runs

### Validation Success Criteria

Successful validation demonstrates:
- **Render Correctness**: GNN spec produces valid Julia with genuine
  `@model pomdp_model(y, A, B, D, u, T)` using `Categorical` / `DiscreteTransition`
- **Inference Convergence**: `infer()` with `free_energy = true` converges
  (validated via `inference_converged` and `vfe_present`)
- **End-to-End Functionality**: Complete pipeline from GNN model to RxInfer.jl simulation
- **Real VFE**: `variational_free_energy` is populated with genuine values
  (previously `Float64[]`); EFE and policy selection remain custom logic
- **Reproducibility**: Same seed yields byte-identical results

## Integration with Pipeline

This documentation is integrated with the 25-step GNN processing pipeline:

1. **Core Processing** (Steps 0-9): GNN parsing, validation, export
   - GNN models parsed and validated
   - RxInfer.jl code generation (Step 11: Render)

2. **Simulation** (Steps 10-16): Model execution and analysis
   - RxInfer.jl execution (Step 12: Execute)
   - Results processing and analysis

3. **Integration** (Steps 17-24): System coordination and output
   - RxInfer.jl results integrated into comprehensive outputs
   - Multi-agent coordination and analysis

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

## Render → Execute → Log → Visualize lifecycle

The RxInfer.jl lifecycle moves a GNN spec through four stages, producing distinct
artifact types at each step. All 29 exemplar GNN files under `input/gnn_files/**`
render to and execute under RxInfer.jl (29/29 render + execute), dispatched by
detected ModelKind to native flat/hierarchical/factored/continuous/learning
generators (multi-agent renders as the documented joint composition).

### 1. Render (Step 11)

`src/render/rxinfer/` consumes `canonical_pomdp_v1` specs and emits an executable
RxInfer.jl script per model:

```text
output/11_render_output/<model>/rxinfer/<model>_rxinfer.jl
```

### 2. Execute (Step 12)

`src/execute/rxinfer/` runs the rendered `.jl` under Julia with RxInfer.jl. The
script writes the **required** result artifact in its working directory:

```text
<model>/rxinfer/simulation_data/simulation_results.json   # rxinfer_simulation_v1
```

### 3. Log (Step 12, best-effort)

The same executed script optionally writes structured runtime logging, guarded so
it never fails the run:

```text
simulation.log          # human-readable chronological trace
simulation_log.json     # machine-readable structured events
```

### 4. Visualize (Steps 12 + 16)

Two complementary visualization layers exist:

- **Julia-native (best-effort, emitted at render/execute time)** — `Plots.jl`
  figures when Plots rendering is available:
  `belief_evolution.png`, `efe_over_time.png`, `policy_posterior.png`.
- **Step-16 matplotlib analysis** — `src/analysis/rxinfer/` produces the full
  per-exemplar set from `rxinfer_simulation_v1`, written under
  `output/16_analysis_output/rxinfer/`: `belief_evolution`, `obs_vs_true`,
  `belief_heatmap`, `belief_entropy`, `accuracy`, `action_frequencies`,
  `belief_convergence`, `belief_trace`, `free_energy`, and `observations`.

All log and visualization artifacts are best-effort and preserve the fields in
`rxinfer_simulation_v1`; only `simulation_results.json` is required.

## Usage Examples

### Running Validation

```bash
# Ensure Julia environment with required packages
julia doc/rxinfer/Multiagent_GNN_RxInfer.jl
```

### Basic RxInfer.jl Model

GNN models are translated to RxInfer.jl factor graphs:

```julia
using RxInfer

# GNN model translated to RxInfer.jl
@model function gnn_model(observations, actions)
    # Hidden state beliefs
    s_f0 ~ Categorical(prior)
    
    # Observations
    o_m0 ~ Categorical(A * s_f0)
    
    # State transitions
    s_f0_next ~ Categorical(B[s_f0, actions])
    
    return s_f0, o_m0, s_f0_next
end
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Framework Integration](../gnn/integration/framework_integration_guide.md)**: Framework integration overview

### Framework Integration
- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[DisCoPy Integration](../discopy/gnn_discopy.md)**: Category theory framework
- **[Execution Guide](../execution/README.md)**: Framework execution strategies

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with Julia and RxInfer.jl foundations
- **Functionality**: Describes actual RxInfer.jl integration capabilities
- **Completeness**: Comprehensive coverage of RxInfer.jl integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[RxInfer Integration](../CROSS_REFERENCE_INDEX.md#rxinferjl)**: Cross-reference index entry
- **[Framework Integration](../gnn/integration/framework_integration_guide.md)**: Framework integration overview
- **[Multi-agent Systems](../gnn/advanced/gnn_multiagent.md)**: Multi-agent modeling
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new RxInfer.jl features and integration capabilities
